
import yaml
import os
import argparse
import train
import visualize_results
import pandas as pd
from datetime import datetime
from utils.reproducibility import set_seed

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="Run Ore Grade Prediction Pipeline (Multi-Model Comparison)")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to configuration file')
    args = parser.parse_args()

    print(f"=== Starting Multi-Model Pipeline with config: {args.config} ===")
    
    # Set Global Seed
    set_seed(42)
    
    # Load Config
    config = load_config(args.config)

    # Parse Model Selection
    model_config = config.get('model_selection', {})
    available_models = model_config.get('available_models', {
        1: 'resnet18', 2: 'mobilenet_v3_small', 3: 'efficientnet_b0'
    })
    selection = model_config.get('selected_models', 'all')
    
    models_to_test = []
    
    if str(selection).lower() == 'all':
        models_to_test = list(available_models.values())
    else:
        # Handle formats: 1 or "1" or "1,2" or [1, 2]
        if isinstance(selection, (int, str)) and ',' not in str(selection) and '[' not in str(selection):
             # Single ID
             ids = [int(selection)]
        elif isinstance(selection, list):
            ids = [int(x) for x in selection]
        elif isinstance(selection, str):
            # Parse commas
            clean_sel = selection.replace('[','').replace(']','')
            ids = [int(x.strip()) for x in clean_sel.split(',')]
        else:
            ids = []
            
        # Map IDs to names
        for mid in ids:
            if mid in available_models:
                models_to_test.append(available_models[mid])
            else:
                print(f"Warning: Model ID {mid} not found in available_models.")

    if not models_to_test:
        print("No valid models selected to run!")
        return
        
    print(f"Models selected for execution: {models_to_test}")
    
    all_results = {}
    
    for model_name in models_to_test:
        print(f"\n" + "="*50)
        print(f"Processing Model: {model_name}")
        print("="*50)
        
        try:
            print(f"\n--- Step 1: Training ({model_name}) ---")
            train.main(config_path=args.config, model_name=model_name)
            
            print(f"\n--- Step 2: Visualization & Evaluation ({model_name}) ---")
            metrics = visualize_results.main(config_path=args.config, model_name=model_name)
            
            if metrics:
                all_results[model_name] = metrics
        except Exception as e:
            print(f"!!! Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*50)
    print("FINAL COMPARISON RESULTS")
    print("="*50)
    
    # Collect data for DataFrame
    # Structure: Model | Target | MSE | MAE | R2
    rows = []
    for model_name, metrics in all_results.items():
        for target, scores in metrics.items():
            row = {'Model': model_name, 'Target': target}
            row.update(scores)
            rows.append(row)
            
    if rows:
        df = pd.DataFrame(rows)
        # Reorder columns
        # Reorder columns dynamically
        cols = ['Model', 'Target', 'MSE', 'MAE', 'R2', 'Acc%', 'Recall%', 'W_Acc%', 'W_Recall%', 'Recovery%', 'Yield%', 'Tail_Grade']
        existing_cols = [c for c in cols if c in df.columns]
        df = df[existing_cols]
        
        print("\nDetailed Metrics:")
        print(df.to_string(index=False, float_format="%.2f"))
        
        # Calculate average R2 per model
        avg_r2 = df.groupby('Model')['R2'].mean().sort_values(ascending=False)
        print("\nAverage R2 Score per Model (Higher is better):")
        print(avg_r2)
        
        # Save to file
        config = load_config(args.config)
        exp_name = config.get('experiment_name', 'default_exp')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"results_{exp_name}_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(df.to_string(index=False, float_format="%.2f"))
            
        print(f"\nResults saved to: {filepath}")
    else:
        print("No results collected.")
        
    print("\n=== Pipeline Complete ===")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
