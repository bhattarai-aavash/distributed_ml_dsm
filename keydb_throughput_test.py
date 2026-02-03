#!/usr/bin/env python3
"""
Simple KeyDB throughput test - sets 10 values in 1 second
"""

import redis
import time
import threading
from datetime import datetime

def test_keydb_throughput():
    """Test KeyDB throughput by setting 10 values every 1 second continuously"""
    
    # Connect to KeyDB
    r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    
    print("🚀 KeyDB Continuous Throughput Test")
    print("=" * 50)
    print("Setting 10 values every 1 second...")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    # Test basic connection
    try:
        r.ping()
        print("✅ Connected to KeyDB successfully")
    except Exception as e:
        print(f"❌ Failed to connect to KeyDB: {e}")
        return
    
    # Statistics tracking
    total_operations = 0
    total_time = 0
    start_time = time.time()
    
    try:
        while True:
            cycle_start = time.time()
            current_time = datetime.now()
            
            print(f"\n⏰ Cycle at {current_time.strftime('%H:%M:%S')}")
            
            # Set 10 values
            for i in range(10):
                key = f"continuous_test_{i}"
                value = f"value_{current_time.strftime('%H:%M:%S.%f')}_{i}"
                r.set(key, value)
                total_operations += 1
            
            cycle_end = time.time()
            cycle_duration = cycle_end - cycle_start
            
            # Calculate statistics
            total_time = cycle_end - start_time
            avg_ops_per_sec = total_operations / total_time if total_time > 0 else 0
            current_ops_per_sec = 10 / cycle_duration if cycle_duration > 0 else 0
            
            print(f"  ✅ Set 10 values in {cycle_duration:.3f}s")
            print(f"  📊 Current rate: {current_ops_per_sec:.1f} ops/sec")
            print(f"  📈 Total: {total_operations} operations in {total_time:.1f}s")
            print(f"  📊 Average rate: {avg_ops_per_sec:.1f} ops/sec")
            
            # Sleep to maintain 1-second intervals
            sleep_time = 1.0 - cycle_duration
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"  ⚠️  Cycle took {cycle_duration:.3f}s (longer than 1s target)")
                
    except KeyboardInterrupt:
        print(f"\n\n🛑 Test stopped by user")
        print("=" * 50)
        print("📊 Final Statistics:")
        print(f"  Total operations: {total_operations}")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Average rate: {total_operations/total_time:.1f} operations/second")
        
        # Show KeyDB info
        print(f"\n📈 KeyDB Server Info:")
        info = r.info()
        print(f"  Total commands processed: {info.get('total_commands_processed', 'Unknown')}")
        print(f"  Instantaneous ops/sec: {info.get('instantaneous_ops_per_sec', 'Unknown')}")
        print(f"  Used memory: {info.get('used_memory_human', 'Unknown')}")
        
        # Show slow log
        print(f"\n🐌 Slow Log (last 5 entries):")
        slowlog = r.slowlog_get(5)
        if slowlog:
            for entry in slowlog:
                print(f"  {entry['duration']}μs: {entry['command']}")
        else:
            print("  No slow log entries found")
        
        print("\n🎉 Continuous throughput test completed!")

if __name__ == "__main__":
    test_keydb_throughput()
