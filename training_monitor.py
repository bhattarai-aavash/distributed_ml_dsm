#!/usr/bin/env python3
"""
Training Monitor Program
Run this on your local machine to monitor the training status of all distributed servers
"""

import redis
import json
import time
import pandas as pd
from datetime import datetime
import argparse
import sys

class TrainingMonitor:
    """Monitor training status across all distributed servers"""
    
    def __init__(self, keydb_host='localhost', keydb_port=6379, keydb_db=0, experiment_name='amber_mlp'):
        """Initialize connection to KeyDB"""
        self.redis_client = redis.Redis(
            host=keydb_host, 
            port=keydb_port, 
            db=keydb_db,
            decode_responses=False
        )
        self.experiment_name = experiment_name
    
    def get_all_training_status(self) -> dict:
        """Get training status of all servers"""
        try:
            all_status = {}
            for key in self.redis_client.keys(f"{self.experiment_name}:training_status:*"):
                server_id = key.decode('utf-8').split(':')[-1]
                status_data = self.redis_client.get(key)
                if status_data:
                    all_status[server_id] = json.loads(status_data.decode('utf-8'))
            return all_status
        except Exception as e:
            print(f"❌ Error getting status: {e}")
            return {}
    
    def check_training_completion(self) -> bool:
        """Check if all servers have completed training"""
        try:
            all_status = self.get_all_training_status()
            if not all_status:
                return False
            
            # Check if all servers are completed
            all_completed = all(
                status['status'] in ['completed', 'failed', 'stopped'] 
                for status in all_status.values()
            )
            
            return all_completed
        except Exception as e:
            print(f"❌ Error checking completion: {e}")
            return False
    
    def display_status(self, detailed: bool = False):
        """Display current training status"""
        all_status = self.get_all_training_status()
        
        if not all_status:
            print("❌ No training status found. Are the servers running?")
            return
        
        print(f"\n📊 Training Status Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Summary
        status_counts = {}
        for status_info in all_status.values():
            status = status_info.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print("�� Summary:")
        for status, count in status_counts.items():
            emoji = {
                'running': '🟢',
                'completed': '✅',
                'failed': '❌',
                'stopped': '⏹️',
                'unknown': '❓'
            }.get(status, '❓')
            print(f"  {emoji} {status.capitalize()}: {count}")
        
        # Check completion
        if self.check_training_completion():
            print("\n🎉 All servers have completed training!")
        else:
            running_count = status_counts.get('running', 0)
            if running_count > 0:
                print(f"\n🔄 {running_count} server(s) still running...")
            else:
                print("\n⚠️  Some servers may have issues")
        
        # Detailed status for each server
        if detailed:
            print(f"\n📋 Detailed Status:")
            print("-" * 80)
            
            for server_id, status_info in sorted(all_status.items()):
                status = status_info.get('status', 'unknown')
                iteration = status_info.get('iteration', 0)
                current_loss = status_info.get('current_loss', 0.0)
                current_accuracy = status_info.get('current_accuracy', 0.0)
                last_update = status_info.get('last_update', 'unknown')
                start_time = status_info.get('start_time', 'unknown')
                
                emoji = {
                    'running': '🟢',
                    'completed': '✅',
                    'failed': '❌',
                    'stopped': '⏹️',
                    'unknown': '❓'
                }.get(status, '❓')
                
                print(f"\n{emoji} Server: {server_id}")
                print(f"   Status: {status.upper()}")
                print(f"   Iteration: {iteration}")
                print(f"   Current Loss: {current_loss:.6f}")
                print(f"   Current Accuracy: {current_accuracy:.4f}")
                print(f"   Last Update: {last_update}")
                print(f"   Start Time: {start_time}")
    
    def export_status_to_csv(self, filename: str = None):
        """Export training status to CSV file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_status_{timestamp}.csv"
        
        all_status = self.get_all_training_status()
        if not all_status:
            print("❌ No status data to export")
            return
        
        # Convert to DataFrame
        status_list = []
        for server_id, status_info in all_status.items():
            status_info['server_id'] = server_id
            status_list.append(status_info)
        
        df = pd.DataFrame(status_list)
        
        # Reorder columns
        columns = ['server_id', 'status', 'iteration', 'current_loss', 'current_accuracy', 
                  'last_update', 'start_time']
        df = df.reindex(columns=columns)
        
        # Export
        df.to_csv(filename, index=False)
        print(f"✅ Status exported to {filename}")
    
    def wait_for_completion(self, check_interval: int = 30, max_wait: int = None):
        """Wait for all servers to complete training"""
        print(f"⏳ Waiting for all servers to complete training...")
        print(f"   Checking every {check_interval} seconds")
        if max_wait:
            print(f"   Maximum wait time: {max_wait} seconds")
        print()
        
        start_time = time.time()
        iteration = 0
        
        while True:
            iteration += 1
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Check if max wait time exceeded
            if max_wait and elapsed > max_wait:
                print(f"⏰ Maximum wait time ({max_wait}s) exceeded!")
                break
            
            # Display current status
            print(f"�� Check #{iteration} - Elapsed: {elapsed:.0f}s")
            self.display_status(detailed=False)
            
            # Check if all completed
            if self.check_training_completion():
                print(f"\n🎉 All servers completed training in {elapsed:.0f} seconds!")
                break
            
            # Wait before next check
            if iteration < 1000:  # Prevent infinite loop
                time.sleep(check_interval)
            else:
                print("⚠️  Too many iterations, stopping...")
                break
        
        # Final detailed status
        print(f"\n📋 Final Status:")
        self.display_status(detailed=True)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Monitor distributed MLP training status')
    parser.add_argument('--host', default='localhost', help='KeyDB host (default: localhost)')
    parser.add_argument('--port', type=int, default=6379, help='KeyDB port (default: 6379)')
    parser.add_argument('--db', type=int, default=0, help='KeyDB database (default: 0)')
    parser.add_argument('--experiment', default='amber_mlp', help='Experiment name (default: amber_mlp)')
    parser.add_argument('--detailed', action='store_true', help='Show detailed status')
    parser.add_argument('--export', help='Export status to CSV file')
    parser.add_argument('--wait', type=int, help='Wait for completion, checking every N seconds')
    parser.add_argument('--max-wait', type=int, help='Maximum wait time in seconds')
    
    args = parser.parse_args()
    
    # Initialize monitor
    try:
        monitor = TrainingMonitor(
            keydb_host=args.host,
            keydb_port=args.port,
            keydb_db=args.db,
            experiment_name=args.experiment
        )
    except Exception as e:
        print(f"❌ Failed to connect to KeyDB: {e}")
        sys.exit(1)
    
    # Export if requested
    if args.export:
        monitor.export_status_to_csv(args.export)
        return
    
    # Wait for completion if requested
    if args.wait:
        monitor.wait_for_completion(check_interval=args.wait, max_wait=args.max_wait)
        return
    
    # Display current status
    monitor.display_status(detailed=args.detailed)

if __name__ == "__main__":
    main()