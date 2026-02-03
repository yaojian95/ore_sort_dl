
import os
# Fix OpenMP runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import glob
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, recall_score

from data.dataset import OreDataset
from models.network import get_model

def calculate_sorting_metrics(targets, preds, weights, threshold):
    """
    Calculate Recovery, Yield, and Tailings Grade based on a classification threshold.
    """
    total_mass = np.sum(weights)
    total_metal = np.sum(targets * weights)
    
    # Decisions: 1 = Accept (Concentrate), 0 = Reject (Waste)
    decisions = (preds >= threshold).astype(int)
    
    # Mass of accepted and rejected rocks
    concentrate_mass = np.sum(weights[decisions == 1])
    waste_mass = np.sum(weights[decisions == 0])
    
    # Metal mass in concentrate (Recovery numerator)
    concentrate_metal = np.sum(targets[decisions == 1] * weights[decisions == 1])
    
    if total_metal > 0:
        recovery = (concentrate_metal / total_metal) * 100
    else:
        recovery = 0.0
        
    if total_mass > 0:
        yield_waste = (waste_mass / total_mass) * 100 # Rejection Rate
    else:
        yield_waste = 0.0
        
    # Tailings Grade
    if waste_mass > 0:
        tailings_grade = np.sum(targets[decisions == 0] * weights[decisions == 0]) / waste_mass
    else:
        tailings_grade = 0.0 # No tailings
        
    return recovery, yield_waste, tailings_grade

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main(config_path='configs/config.yaml', model_name='resnet18'):
    # config_path is passed as argument
    config = load_config(config_path)
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')

    # Load Data (Same seed for consistent split)
    data_transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
    ])
    
    full_dataset = OreDataset(
        pickle_path=config['data']['pickle_path'],
        target_columns=config['data']['target_columns'],
        transform=data_transform
    )
    
    total_size = len(full_dataset)
    test_size = int(total_size * config['data']['test_split'])
    val_size = int(total_size * config['data']['val_split'])
    train_size = total_size - test_size - val_size

    _, _, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    test_loader = DataLoader(test_dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])

    # Find latest model
    exp_name = config.get('experiment_name', 'default_exp')
    save_dir = os.path.join(config['training']['save_dir'], exp_name, model_name)
    
    if not os.path.exists(save_dir):
        print(f"Directory not found: {save_dir}")
        return None

    list_of_files = glob.glob(os.path.join(save_dir, '*.pth'))
    if not list_of_files:
        print(f"No model found in {save_dir}!")
        return None
        
    # Prefer 'best_model.pth' if exists, else latest
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        latest_model_path = best_model_path
    else:
        latest_model_path = max(list_of_files, key=os.path.getctime)
        
    print(f"Loading model: {latest_model_path}")

    # Load Model
    model = get_model(
        model_name=model_name,
        in_channels=config['model']['in_channels'],
        num_targets=len(config['data']['target_columns']),
        pretrained=False # Weights loaded from file
    ).to(device)
    model.load_state_dict(torch.load(latest_model_path, map_location=device))
    model.eval()

    # Inference
    all_preds = []
    all_targets = []
    all_weights = []
    
    with torch.no_grad():
        for images, targets, weights in test_loader:
            images = images.to(device)
            outputs = model(images)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())
            all_weights.append(weights.numpy())
            
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    all_weights = np.concatenate(all_weights)

    # Metrics & Plotting
    target_names = config['data']['target_columns']
    
    metrics = {}
    print(f"\nEvaluation Results for {model_name}:")

    
    thresholds = config.get('metrics', {}).get('thresholds', {})
    
    # 1. Individual Target Metrics
    for i, name in enumerate(target_names):
        # Regression Metrics
        mse = mean_squared_error(all_targets[:, i], all_preds[:, i])
        mae = mean_absolute_error(all_targets[:, i], all_preds[:, i])
        r2 = r2_score(all_targets[:, i], all_preds[:, i])
        
        
        metrics[name] = {'MSE': round(mse, 2), 'MAE': round(mae, 2), 'R2': round(r2, 2)}
        print(f"{name}: MSE={mse:.2f}, MAE={mae:.2f}, R2={r2:.2f}")
        
        # Classification Metrics (if threshold exists)
        if name in thresholds:
            thresh = thresholds[name]
            # Binarize
            true_cls = (all_targets[:, i] >= thresh).astype(int)
            pred_cls = (all_preds[:, i] >= thresh).astype(int)
            
            acc = accuracy_score(true_cls, pred_cls) * 100
            rec = recall_score(true_cls, pred_cls, zero_division=0) * 100
            
            # Weighted Metrics
            w_acc = accuracy_score(true_cls, pred_cls, sample_weight=all_weights) * 100
            w_rec = recall_score(true_cls, pred_cls, sample_weight=all_weights, zero_division=0) * 100
            
            # Sorting Metrics
            recov, yield_w, tail_g = calculate_sorting_metrics(all_targets[:, i], all_preds[:, i], all_weights, thresh)
            
            metrics[name].update({
                'Acc%': round(acc, 2), 'Recall%': round(rec, 2), 
                'W_Acc%': round(w_acc, 2), 'W_Recall%': round(w_rec, 2),
                'Recovery%': round(recov, 2), 'Yield%': round(yield_w, 2), 'Tail_Grade': round(tail_g, 2)
            })
            print(f"  [Classify > {thresh}]: Acc={acc:.2f}, Rec={rec:.2f} | W_Acc={w_acc:.2f}, W_Rec={w_rec:.2f}")
            print(f"    -> Recovery={recov:.2f}%, Waste Yield={yield_w:.2f}%, Tail Grade={tail_g:.2f}")

    # 2. Composite Metrics
    # Helper to find index safely
    def get_idx(col_name):
        try:
            return target_names.index(col_name)
        except ValueError:
            return None

    # Pb + Zn
    idx_Pb = get_idx('Pb_grade')
    idx_Zn = get_idx('Zn_grade')
    if idx_Pb is not None and idx_Zn is not None and 'Pb_Zn_grade' in thresholds:
        thresh = thresholds['Pb_Zn_grade']
        true_sum = all_targets[:, idx_Pb] + all_targets[:, idx_Zn]
        pred_sum = all_preds[:, idx_Pb] + all_preds[:, idx_Zn]
        
        true_cls = (true_sum >= thresh).astype(int)
        pred_cls = (pred_sum >= thresh).astype(int)
        
        # Regression Metrics for Composite
        mse = mean_squared_error(true_sum, pred_sum)
        mae = mean_absolute_error(true_sum, pred_sum)
        r2 = r2_score(true_sum, pred_sum)
        
        acc = accuracy_score(true_cls, pred_cls) * 100
        rec = recall_score(true_cls, pred_cls, zero_division=0) * 100
        w_acc = accuracy_score(true_cls, pred_cls, sample_weight=all_weights) * 100
        w_rec = recall_score(true_cls, pred_cls, sample_weight=all_weights, zero_division=0) * 100
        
        # Sorting Metrics
        recov, yield_w, tail_g = calculate_sorting_metrics(true_sum, pred_sum, all_weights, thresh)
        
        metrics['Pb+Zn'] = {
            'MSE': round(mse, 2), 'MAE': round(mae, 2), 'R2': round(r2, 2),
            'Acc%': round(acc, 2), 'Recall%': round(rec, 2), 
            'W_Acc%': round(w_acc, 2), 'W_Recall%': round(w_rec, 2),
            'Recovery%': round(recov, 2), 'Yield%': round(yield_w, 2), 'Tail_Grade': round(tail_g, 2)
        }
        print(f"Pb+Zn (Reg/Classify > {thresh}): MSE={mse:.2f}, MAE={mae:.2f}, R2={r2:.2f} | Acc={acc:.2f}, Rec={rec:.2f} | W_Acc={w_acc:.2f}, W_Rec={w_rec:.2f}")
        print(f"    -> Recovery={recov:.2f}%, Waste Yield={yield_w:.2f}%, Tail Grade={tail_g:.2f}")

    # Pb + Zn + Fe
    idx_Fe = get_idx('Fe_grade')
    if idx_Pb is not None and idx_Zn is not None and idx_Fe is not None and 'Pb_Zn_Fe_grade' in thresholds:
        thresh = thresholds['Pb_Zn_Fe_grade']
        true_sum = all_targets[:, idx_Pb] + all_targets[:, idx_Zn] + all_targets[:, idx_Fe]
        pred_sum = all_preds[:, idx_Pb] + all_preds[:, idx_Zn] + all_preds[:, idx_Fe]
        
        true_cls = (true_sum >= thresh).astype(int)
        pred_cls = (pred_sum >= thresh).astype(int)
        
        # Regression Metrics for Composite
        mse = mean_squared_error(true_sum, pred_sum)
        mae = mean_absolute_error(true_sum, pred_sum)
        r2 = r2_score(true_sum, pred_sum)
        
        acc = accuracy_score(true_cls, pred_cls) * 100
        rec = recall_score(true_cls, pred_cls, zero_division=0) * 100
        w_acc = accuracy_score(true_cls, pred_cls, sample_weight=all_weights) * 100
        w_rec = recall_score(true_cls, pred_cls, sample_weight=all_weights, zero_division=0) * 100
        
        
        # Sorting Metrics
        # recovery, yield_w, tail_g = calculate_sorting_metrics(true_sum, pred_sum, all_weights, thresh)
        # Note: Composite target is a sum, but logic is same
        recov, yield_w, tail_g = calculate_sorting_metrics(true_sum, pred_sum, all_weights, thresh)

        metrics['Pb+Zn+Fe'] = {
            'MSE': round(mse, 2), 'MAE': round(mae, 2), 'R2': round(r2, 2),
            'Acc%': round(acc, 2), 'Recall%': round(rec, 2), 
            'W_Acc%': round(w_acc, 2), 'W_Recall%': round(w_rec, 2),
            'Recovery%': round(recov, 2), 'Yield%': round(yield_w, 2), 'Tail_Grade': round(tail_g, 2)
        }
        print(f"Pb+Zn+Fe (Reg/Classify > {thresh}): MSE={mse:.2f}, MAE={mae:.2f}, R2={r2:.2f} | Acc={acc:.2f}, Rec={rec:.2f} | W_Acc={w_acc:.2f}, W_Rec={w_rec:.2f}")
        print(f"    -> Recovery={recov:.2f}%, Waste Yield={yield_w:.2f}%, Tail Grade={tail_g:.2f}")

    # Plot Scatter
    fig, axes = plt.subplots(1, len(target_names), figsize=(5*len(target_names), 5))
    if len(target_names) == 1:
        axes = [axes]
        
    for i, ax in enumerate(axes):
        ax.scatter(all_targets[:, i], all_preds[:, i], alpha=0.5)
        
        # Perfect prediction line
        min_val = min(all_targets[:, i].min(), all_preds[:, i].min())
        max_val = max(all_targets[:, i].max(), all_preds[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{target_names[i]} Prediction')
        
    plt.tight_layout()
    
    log_dir = os.path.join(config['training']['log_dir'], exp_name, model_name)
    os.makedirs(log_dir, exist_ok=True)
    plot_path = os.path.join(log_dir, 'prediction_scatter.png')
    plt.savefig(plot_path)
    print(f"Prediction plots saved to {plot_path}")
    
    return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--model', type=str, default='resnet18')
    args = parser.parse_args()
    
    main(config_path=args.config, model_name=args.model)
