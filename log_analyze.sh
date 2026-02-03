#!/bin/bash

# Get experiment name from command line argument
# if [ $# -eq 0 ]; then
#     echo "Usage: $0 <experiment_name>"
#     echo "Example: $0 cic_mlp_async"
#     exit 1
# fi

EXPERIMENT_NAME="ember_mlp_async"
server_log_dir="/home/abhattar/keydb_logs/ember_mlp_async"

if [ ! -d "$server_log_dir" ]; then
    echo "Error: '$server_log_dir' is not a valid directory."
    exit 1
fi

# Create output directory for results
output_dir="keydb_analysis_${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$output_dir"
echo "📁 Results will be saved to: $output_dir"

# Read all files and parse using AWK


aggregate_clients_detailed() {
    awk '
    {
        timestamp = $4 " " $5
        client = FILENAME
        key = timestamp "|" client
        count[key][$1]++
        clients[client] = 1
        timestamps[timestamp] = 1
        global_count[timestamp][$1]++
    }

    END {
        printf "%-20s %-20s %-10s %-10s %-10s\n", "Timestamp", "Client", "Read", "Write", "Total"
        PROCINFO["sorted_in"] = "@ind_str_asc"
        for (ts in timestamps) {
            for (cl in clients) {
                key = ts "|" cl
                read = count[key]["Read"] + 0
                write = count[key]["Write"] + 0
                total = read + write
                if (total > 0) {
                    printf "%-20s %-20s %-10d %-10d %-10d\n", ts, cl, read, write, total
                }
            }
            # Print total for this timestamp
            global_read = global_count[ts]["Read"] + 0
            global_write = global_count[ts]["Write"] + 0
            global_total = global_read + global_write
            printf "%-20s %-20s %-10d %-10d %-10d\n", ts, "TOTAL", global_read, global_write, global_total
            print "--------------------------------------------------------------------------------"
        }
    }
    ' "$clients_log_dir"/*
} 


aggregate_clients()
{


awk '
{
    timestamp = $4 " " $5
    op = $1
    key = timestamp " " op
    count[key]++
    timestamps[timestamp] = 1
}
END {
    printf "%-20s %-12s %-12s %-12s\n", "Timestamp", "Read", "Write", "Total"
    PROCINFO["sorted_in"] = "@ind_str_asc"
    for (ts in timestamps) {
        read_key = ts " Read"
        write_key = ts " Write"
        read_count = count[read_key] + 0
        write_count = count[write_key] + 0
        total = read_count + write_count
        printf "%-20s %-12d %-12d %-12d\n", ts, read_count, write_count, total
    }
}
' "$clients_log_dir"/*

}
aggregate_server_command_counts() {
    # Process each experiment directory separately
    for exp_dir in "$server_log_dir"/experiment_*/; do
        if [ -d "$exp_dir" ]; then
            # Extract experiment number from directory path
            exp_name=$(basename "$exp_dir")
            echo "Processing $exp_name for experiment: $EXPERIMENT_NAME"
            
            # Find all mteverest*.log files in server subdirectories
            log_files=()
            for server_dir in "$exp_dir"mteverest*/; do
                if [ -d "$server_dir" ]; then
                    for log_file in "$server_dir"mteverest*.log; do
                        if [ -f "$log_file" ]; then
                            log_files+=("$log_file")
                        fi
                    done
                fi
            done
            
            if [ ${#log_files[@]} -eq 0 ]; then
                echo "⚠️  No mteverest*.log files found in $exp_name"
                continue
            fi
            
            echo "Found ${#log_files[@]} log files: ${log_files[*]}"
            
            # Analyze this experiment's server logs
            awk '
            /Current throughput:/ {
                # Construct timestamp without milliseconds
                split($5, tparts, "\\.")   # $5 = 15:53:02.998
                ts_clean = $2 " " $3 " " $4 " " tparts[1]

                # Extract server name from path (e.g., /path/experiment_1/mteverest1/mteverest1.log -> mteverest1)
                n = split(FILENAME, path_parts, "/")
                full_filename = path_parts[n]
                # Remove .log extension and get server name
                gsub(/\\.log$/, "", full_filename)
                server = full_filename

                # Count each occurrence (i.e., each command execution)
                count_by_ts_server[ts_clean "|" server]++
                total_by_ts[ts_clean]++
                servers[server] = 1
                timestamps[ts_clean] = 1
            }
            END {
                printf "%-20s %-20s %-15s\n", "Timestamp", "Server", "CommandCount"
                PROCINFO["sorted_in"] = "@ind_str_asc"
                for (ts in timestamps) {
                    for (srv in servers) {
                        key = ts "|" srv
                        val = count_by_ts_server[key] + 0
                        if (val > 0) {
                            printf "%-20s %-20s %-15d\n", ts, srv, val
                        }
                    }
                    printf "%-20s %-20s %-15d\n", ts, "TOTAL", total_by_ts[ts]
                    print "---------------------------------------------------------------"
                }
            }
            ' "${log_files[@]}" > "$output_dir/server_counts_${EXPERIMENT_NAME}_${exp_name}.txt"
            
            echo "✅ Saved server counts for $exp_name to: $output_dir/server_counts_${EXPERIMENT_NAME}_${exp_name}.txt"
        fi
    done
}

SECONDS=0
aggregate_server_command_counts
# aggregate_clients > client_counts.txt
duration=$SECONDS
echo "Elapsed time: $duration seconds"
