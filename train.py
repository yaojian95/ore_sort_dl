
import os
# Fix OpenMP runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from data.dataset import OreDataset
from models.network import get_model
from utils.reproducibility import set_seed

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main(config_path='configs/config.yaml', model_name='resnet18'):
    # 1. Load Configuration
    # config_path is passed as argument
    config = load_config(config_path)
    
    # Set Seed for Reproducibility
    set_seed(42)
    
    # Setup Device
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Construct paths with experiment name AND model name
    exp_name = config.get('experiment_name', 'default_exp')
    save_dir = os.path.join(config['training']['save_dir'], exp_name, model_name)
    log_dir = os.path.join(config['training']['log_dir'], exp_name, model_name)
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 2. Prepare Data
    # Transform: Resize and Normalize
    # Initial data is [0, 255] (implied), ToTensor scales to [0, 1].
    # Normalization mean/std can be computed, but [0.5, 0.5] is a safe start.
    data_transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(), # Converts PIL [0, 255] to Tensor [0.0, 1.0]
        # transforms.Normalize(mean=[0.5], std=[0.5]) # Optional, if we want -1 to 1
    ])

    full_dataset = OreDataset(
        pickle_path=config['data']['pickle_path'],
        target_columns=config['data']['target_columns'],
        transform=data_transform
    )

    # Split
    total_size = len(full_dataset)
    test_size = int(total_size * config['data']['test_split'])
    val_size = int(total_size * config['data']['val_split'])
    train_size = total_size - test_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True, num_workers=config['data']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])
    # test_loader kept for `test.py` or final eval

    # 3. Model Setup
    print(f"Initializing model: {model_name}")
    model = get_model(
        model_name=model_name,
        in_channels=config['model']['in_channels'],
        num_targets=len(config['data']['target_columns']),
        pretrained=True
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # scheduler: Reduce LR when val_loss stops decreasing
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 4. Training Loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Starting Training for {model_name}...")
    for epoch in range(config['training']['epochs']):
        # Train
        model.train()
        running_loss = 0.0
        for images, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [Train]", leave=False):
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_train_loss = running_loss / len(train_dataset)
        
        # Val
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item() * images.size(0)
                
        epoch_val_loss = val_running_loss / len(val_dataset)
        
        # Step Scheduler
        before_lr = optimizer.param_groups[0]['lr']
        scheduler.step(epoch_val_loss)
        after_lr = optimizer.param_groups[0]['lr']
        if after_lr != before_lr:
           print(f"Epoch {epoch+1}: LR reduced from {before_lr} to {after_lr}")
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss={epoch_train_loss:.4f}, Val Loss={epoch_val_loss:.4f}")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            save_path = os.path.join(save_dir, f'best_model.pth') # Fixed name for easier loading
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved best model to {save_path}")

    # 5. Save History & Final Plot
    # Save training history for visualization
    # import json
    # with open(os.path.join(log_dir, f'history_{timestamp}.json'), 'w') as f:
    #     json.dump(history, f)

    # Simple plot
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title(f'Training Process ({exp_name} - {model_name})')
    plt.legend()
    plt.savefig(os.path.join(log_dir, f'loss_curve_{timestamp}.png'))
    print(f"Training Complete for {model_name}.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--model', type=str, default='resnet18')
    args = parser.parse_args()
    
    main(config_path=args.config, model_name=args.model)
