#!/usr/bin/env python3
"""
Simple Memory Monitor for yangra machines
Run this to monitor memory usage and prevent OOM kills
"""

import psutil
import time
import os
from datetime import datetime

def get_memory_info():
    """Get current memory information"""
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    return {
        'total_mb': memory.total / (1024 * 1024),
        'used_mb': memory.used / (1024 * 1024),
        'available_mb': memory.available / (1024 * 1024),
        'percent': memory.percent,
        'swap_total_mb': swap.total / (1024 * 1024),
        'swap_used_mb': swap.used / (1024 * 1024),
        'swap_percent': swap.percent
    }

def get_process_memory():
    """Get current process memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / (1024 * 1024),
        'vms_mb': memory_info.vms / (1024 * 1024),
        'percent': process.memory_percent()
    }

def monitor_memory(interval=5, threshold=80):
    """Monitor memory usage continuously"""
    print(f"🔍 Memory Monitor Started - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⚠️  Warning threshold: {threshold}%")
    print(f"📊 Monitoring every {interval} seconds")
    print("-" * 80)
    
    try:
        while True:
            # Get memory info
            mem_info = get_memory_info()
            proc_info = get_process_memory()
            
            # Current timestamp
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            # Memory status
            if mem_info['percent'] > threshold:
                status = "🚨 HIGH"
            elif mem_info['percent'] > threshold * 0.8:
                status = "⚠️  WARNING"
            else:
                status = "✅ OK"
            
            # Print status
            print(f"[{timestamp}] {status} | "
                  f"RAM: {mem_info['used_mb']:.0f}MB/{mem_info['total_mb']:.0f}MB "
                  f"({mem_info['percent']:.1f}%) | "
                  f"Available: {mem_info['available_mb']:.0f}MB | "
                  f"Process: {proc_info['rss_mb']:.0f}MB")
            
            # Print swap info if used
            if mem_info['swap_used_mb'] > 0:
                print(f"    💾 SWAP: {mem_info['swap_used_mb']:.0f}MB/{mem_info['swap_total_mb']:.0f}MB "
                      f"({mem_info['swap_percent']:.1f}%)")
            
            # Warning messages
            if mem_info['percent'] > threshold:
                print(f"    🚨 WARNING: Memory usage is {mem_info['percent']:.1f}%!")
                print(f"    💡 Consider reducing batch size or chunk size")
            
            if mem_info['available_mb'] < 1000:  # Less than 1GB available
                print(f"    ⚠️  Low available memory: {mem_info['available_mb']:.0f}MB")
            
            # Wait for next check
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print(f"\n🛑 Memory monitoring stopped - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def quick_check():
    """Quick memory check"""
    mem_info = get_memory_info()
    proc_info = get_process_memory()
    
    print(f"📊 Quick Memory Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"RAM: {mem_info['used_mb']:.0f}MB / {mem_info['total_mb']:.0f}MB ({mem_info['percent']:.1f}%)")
    print(f"Available: {mem_info['available_mb']:.0f}MB")
    print(f"Process: {proc_info['rss_mb']:.0f}MB")
    
    if mem_info['percent'] > 80:
        print("🚨 HIGH MEMORY USAGE!")
    elif mem_info['percent'] > 70:
        print("⚠️  WARNING: Memory usage is getting high")
    else:
        print("✅ Memory usage is normal")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "monitor":
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            threshold = int(sys.argv[3]) if len(sys.argv) > 3 else 80
            monitor_memory(interval, threshold)
        elif sys.argv[1] == "check":
            quick_check()
        else:
            print("Usage:")
            print("  python memory_monitor.py monitor [interval] [threshold]")
            print("  python memory_monitor.py check")
    else:
        quick_check()
