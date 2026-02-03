import pandas as pd
import numpy as np

def debug_average_calculation():
    """Debug the average calculation logic"""
    
    # Load the actual data from the logs
    logs_dir = "/home/abhattar/ml_cars_main/training_logs_20250901_134627"
    
    server_data = {}
    
    # Load data from each server
    for server_dir in ['mteverest1', 'mteverest3', 'mteverest4']:
        server_path = f"{logs_dir}/{server_dir}"
        
        # Find CSV files in server directory
        import os
        csv_files = [f for f in os.listdir(server_path) if f.endswith('.csv')]
        
        if csv_files:
            csv_file = f"{server_path}/{csv_files[0]}"
            print(f"Loading data from {server_dir}: {csv_file}")
            
            # Load CSV data
            df = pd.read_csv(csv_file)
            server_data[server_dir] = df
            print(f"  Loaded {len(df)} rows")
    
    # Test the average calculation for a few iterations
    test_iterations = [1, 100, 500, 1000, 2000]
    
    for iteration in test_iterations:
        print(f"\n=== Testing Iteration {iteration} ===")
        
        valid_servers = 0
        total_time = 0.0
        total_train_loss = 0.0
        total_train_acc = 0.0
        total_val_loss = 0.0
        total_val_acc = 0.0
        
        # Track which metrics we actually have data for
        has_train_loss = False
        has_train_acc = False
        has_val_loss = False
        has_val_acc = False
        
        for server_name, df in server_data.items():
            server_iteration = df[df['iteration'] == iteration]
            
            if not server_iteration.empty:
                valid_servers += 1
                
                # Get metrics for this iteration
                row = server_iteration.iloc[0]
                
                print(f"  {server_name}:")
                print(f"    train_loss: {row['train_loss']} (valid: {pd.notna(row['train_loss'])})")
                print(f"    train_acc: {row['train_acc']} (valid: {pd.notna(row['train_acc'])})")
                print(f"    val_loss: {row['val_loss']} (valid: {pd.notna(row['val_loss'])})")
                print(f"    val_acc: {row['val_acc']} (valid: {pd.notna(row['val_acc'])})")
                
                if pd.notna(row['time_elapsed']):
                    total_time += row['time_elapsed']
                
                if pd.notna(row['train_loss']):
                    total_train_loss += row['train_loss']
                    has_train_loss = True
                
                if pd.notna(row['train_acc']):
                    total_train_acc += row['train_acc']
                    has_train_acc = True
                
                if pd.notna(row['val_loss']):
                    total_val_loss += row['val_loss']
                    has_val_loss = True
                
                if pd.notna(row['val_acc']):
                    total_val_acc += row['val_acc']
                    has_val_acc = True
        
        print(f"  Summary for iteration {iteration}:")
        print(f"    Valid servers: {valid_servers}")
        print(f"    Total train_loss: {total_train_loss}")
        print(f"    Total val_loss: {total_val_loss}")
        print(f"    Has train_loss: {has_train_loss}")
        print(f"    Has val_loss: {has_val_loss}")
        
        if valid_servers > 0:
            avg_train_loss = total_train_loss / valid_servers if has_train_loss else np.nan
            avg_val_loss = total_val_loss / valid_servers if has_val_loss else np.nan
            
            print(f"    Average train_loss: {avg_train_loss}")
            print(f"    Average val_loss: {avg_val_loss}")
            
            # Also calculate the "old way" to compare
            old_avg_train_loss = total_train_loss / valid_servers if total_train_loss > 0 else np.nan
            old_avg_val_loss = total_val_loss / valid_servers if total_val_loss > 0 else np.nan
            
            print(f"    Old way avg train_loss: {old_avg_train_loss}")
            print(f"    Old way avg val_loss: {old_avg_val_loss}")

if __name__ == "__main__":
    debug_average_calculation()
