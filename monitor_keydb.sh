#!/bin/bash

# KeyDB Monitoring Script
# Monitors KeyDB logs for throughput and performance metrics

echo "🔍 KeyDB Monitoring Script"
echo "=========================="

# Check if KeyDB is running
if pgrep -f "keydb-server" > /dev/null; then
    echo "✅ KeyDB server is running"
else
    echo "❌ KeyDB server is not running"
    exit 1
fi

# Function to monitor logs
monitor_logs() {
    echo "📊 Monitoring KeyDB logs (Press Ctrl+C to stop)..."
    echo ""
    
    # Monitor the log file for new entries
    tail -f KeyDB/keydb_log.log | while read line; do
        timestamp=$(date '+%H:%M:%S')
        echo "[$timestamp] $line"
    done
}

# Function to show KeyDB stats
show_stats() {
    echo "📈 KeyDB Statistics:"
    echo "==================="
    
    # Connect to KeyDB and get info
    redis-cli -h localhost -p 6379 info | grep -E "(redis_version|connected_clients|used_memory_human|total_commands_processed|instantaneous_ops_per_sec)"
    
    echo ""
    echo "🐌 Slow Log (last 10 entries):"
    redis-cli -h localhost -p 6379 slowlog get 10
    
    echo ""
    echo "⏱️  Latency Stats:"
    redis-cli -h localhost -p 6379 latency latest
}

# Function to grep for specific patterns
grep_logs() {
    echo "🔍 Searching KeyDB logs for patterns..."
    echo ""
    
    # Search for different types of operations
    echo "📝 SET operations:"
    grep -i "set" KeyDB/keydb_log.log | tail -5
    
    echo ""
    echo "📖 GET operations:"
    grep -i "get" KeyDB/keydb_log.log | tail -5
    
    echo ""
    echo "⚠️  Warnings/Errors:"
    grep -i "warning\|error\|fail" KeyDB/keydb_log.log | tail -5
    
    echo ""
    echo "🚀 Performance metrics:"
    grep -i "ops\|throughput\|latency" KeyDB/keydb_log.log | tail -5
}

# Main menu
case "$1" in
    "monitor")
        monitor_logs
        ;;
    "stats")
        show_stats
        ;;
    "grep")
        grep_logs
        ;;
    *)
        echo "Usage: $0 {monitor|stats|grep}"
        echo ""
        echo "Commands:"
        echo "  monitor  - Monitor KeyDB logs in real-time"
        echo "  stats    - Show KeyDB statistics and slow log"
        echo "  grep     - Search logs for specific patterns"
        echo ""
        echo "Examples:"
        echo "  $0 monitor    # Watch logs live"
        echo "  $0 stats      # Show current stats"
        echo "  $0 grep       # Search for patterns"
        ;;
esac



