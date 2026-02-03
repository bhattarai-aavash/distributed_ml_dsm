#!/bin/bash

# Analyze KeyDB CMD_LOG entries to compute per-server throughput broken down by source
# Usage: ./separate_log_analyzer.sh <server_log_dir>
#
# Directory structure: Recursively searches subdirectories for mteverest*.log files
# Example: server_log_dir/
#          ├── mteverest1/mteverest1.log
#          ├── mteverest3/mteverest3.log
#          └── mteverest4/mteverest4.log
#
# UPDATED: Now uses the context field from CMD_LOG to accurately identify command sources:
#   - direct_client: Commands from direct client connections (source = "client")
#   - replicated_from_master: Commands replicated from other masters (source = normalized master hostname)
#   - replicated_to_replica: Commands being propagated to replicas (IGNORED - not executed here)
#
# The peer field for replicated commands now contains the actual master hostname/IP (thanks to UUID fallback),
# so we can accurately track which master originated each replicated command.

server_log_dir="/home/abhattar/Desktop/research/Distributed_Project/keydb_logs"
# Allow overriding the log dir via CLI args (handles paths with spaces)
if [ $# -gt 0 ]; then
    # Join all arguments with spaces (handles unquoted paths with spaces)
    server_log_dir="$*"
fi

# Verify the directory exists
if [ ! -d "$server_log_dir" ]; then
    echo "❌ Error: Directory does not exist: $server_log_dir"
    echo "💡 Tip: If your path contains spaces, make sure to quote it:"
    echo "   ./separate_log_analyzer.sh \"/home/abhattar/Keydb logs/ember_mlp_async\""
    exit 1
fi

timestamp_dir=$(date +%Y%m%d_%H%M%S)
output_dir="cmd_throughput_${timestamp_dir}"
mkdir -p "$output_dir"
echo "📁 Results will be saved to: $output_dir"
echo "🔍 Searching for mteverest*.log files in: $server_log_dir"

found_any=false

# Look for mteverest*.log files recursively in subdirectories
log_files=()
while IFS= read -r -d '' log_file; do
    log_files+=("$log_file")
done < <(find "$server_log_dir" -type f -name "mteverest*.log" -print0 2>/dev/null)

if [ ${#log_files[@]} -eq 0 ]; then
    echo "⚠️  No mteverest*.log files found in: $server_log_dir"
    echo "💡 Make sure the directory path is correct and contains subdirectories with mteverest*.log files"
    exit 1
fi

echo "📊 Found ${#log_files[@]} log files:"
for log_file in "${log_files[@]}"; do
    # Show relative path from server_log_dir for clarity
    rel_path="${log_file#$server_log_dir/}"
    echo "  - $rel_path"
done

exp_name="keydb_logs"
out_csv="$output_dir/throughput_${exp_name}.csv"
summary_csv="$output_dir/throughput_summary_${exp_name}.csv"
per_server_dir="$output_dir/${exp_name}"
mkdir -p "$per_server_dir"
echo "experiment,server,source,second,count" > "$out_csv"
echo "experiment,server,second,client_count,replicated_count,total_count" > "$summary_csv"

awk -v exp_name="$exp_name" -v per_server_dir="$per_server_dir" -v summary_csv="$summary_csv" '
function basename(path,   n, a) {
    n = split(path, a, "/");
    return a[n];
}
function without_ext(name) {
    sub(/\.[^.]*$/, "", name);
    return name;
}
# Normalize a hostname or IP: strip brackets, ports, and domain suffixes; lowercase
# For IP addresses, return the full IP (without port) - use IP directly as identifier
# For hostnames, return the first label before dot
function normalize_host(s) {
    s = tolower(s)
    gsub(/^\[/, "", s); gsub(/\]$/, "", s);
    # Remove port
    port_removed = s;
    sub(/:[0-9]+$/, "", port_removed);
    # Check if it is an IP address (IPv4 or IPv6)
    # IPv4: digits.digits.digits.digits (using simpler pattern without quantifiers)
    # IPv6: contains colons and hex digits
    # Use a simpler approach: count dots for IPv4, or check for colons for IPv6
    dot_count = 0;
    for (i = 1; i <= length(port_removed); i++) {
        if (substr(port_removed, i, 1) == ".") dot_count++;
    }
    # IPv4: has exactly 3 dots and all parts are digits
    if (dot_count == 3 && port_removed ~ /^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/) {
        return port_removed;
    }
    # IPv6: contains colons (and possibly hex digits)
    if (port_removed ~ /:/) {
        return port_removed;
    }
    # It is a hostname - take first label before dot
    sub(/\..*$/, "", port_removed);
    return port_removed;
}
# Return 1 if string looks like an IP (IPv4 or IPv6), optionally with port
function is_ip(s) {
    # Remove port and brackets for checking
    temp = s;
    gsub(/^\[/, "", temp); gsub(/\]$/, "", temp);
    sub(/:[0-9]+$/, "", temp);
    # IPv4: count dots
    dot_count = 0;
    for (i = 1; i <= length(temp); i++) {
        if (substr(temp, i, 1) == ".") dot_count++;
    }
    if (dot_count == 3 && temp ~ /^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/) return 1;
    # IPv6: contains colons
    if (temp ~ /:/) return 1;
    return 0;
}
# Simple bubble sort for array - compatible with all awk implementations
function sort_array(arr, n,    i, j, temp) {
    for (i = 1; i <= n; i++) {
        for (j = i + 1; j <= n; j++) {
            if (arr[i] > arr[j]) {
                temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
    }
}
index($0, "CMD_LOG,") {
    # Extract record after CMD_LOG,
    pos = index($0, "CMD_LOG,");
    rec = substr($0, pos + 8);
    # Fields: ts_ms, client_id, role, peer, command, context
    # context can be: direct_client | replicated_from_master | replicated_to_replica
    split(rec, arr, ",");
    if (length(arr) < 5) next;
    ts_ms = arr[1] + 0;
    role = arr[3];
    peer = arr[4];
    # New context field (6th field)
    context = (length(arr) >= 6 ? arr[6] : "");
    second = int(ts_ms / 1000);
    
    # Determine server from filename (basename without .log)
    fname = basename(FILENAME);
    server = without_ext(fname);
    
    # NEW LOGIC: Use context field to determine source
    # Skip entries that are being propagated to replicas (not executed here)
    if (context == "replicated_to_replica") {
        next;
    }
    
    # Determine source based on context field
    if (context == "direct_client") {
        # Direct client command - source is always "client"
        source = "client";
    } else {
        if (context == "replicated_from_master") {
            # Replicated command - dynamically extract master IP/hostname from peer field
            # The peer field contains the actual master IP or hostname from the log
            # Normalize the peer to get a consistent identifier (IP without port, or hostname)
            source = normalize_host(peer);
            # If normalization fails or peer is "unknown_master", use fallback
            if (source == "" || source == "unknown" || peer == "unknown_master") {
                source = "unknown_master";
            }
        } else {
            # Fallback for old-format logs without context field
            # IMPORTANT: Cannot use is_ip() heuristic anymore since both clients and masters have IPs
            # Instead, rely on role field: if role is "master" or "replica", it is replicated
            # Otherwise, assume it is a client
            if (role == "master" || role == "replica") {
                # Replicated command - normalize peer (could be IP or hostname)
                source = normalize_host(peer);
                if (source == "" || source == "unknown") {
                    source = "unknown_master";
                }
            } else {
                # Direct client command
                source = "client";
            }
        }
    }
    
    key = server "|" source "|" second;
    count[key]++;
    
    # Aggregate summary counts by classification
    ksum = server "|" second;
    if (source == "client") {
        client_sum[ksum]++;
    } else {
        repl_sum[ksum]++;
    }
    
    # Track servers list and seconds per server for later pivot output
    servers[server] = 1;
    secs[server "|" second] = 1;
    sources[source] = 1;
    
    # Track min second per server to compute elapsed time
    if ((server in minsec) == 0 || second < minsec[server]) minsec[server] = second;
}
END {
    for (k in count) {
        split(k, p, "|");
        server = p[1]; source = p[2]; second = p[3];
        # Write to aggregate CSV
        printf "%s,%s,%s,%s,%d\n", exp_name, server, source, second, count[k];
        # Also write per-server CSV
        f = per_server_dir "/" server ".csv";
        if (!(f in init)) {
            print "experiment,server,source,second,count" > f;
            init[f] = 1;
        }
        printf "%s,%s,%s,%s,%d\n", exp_name, server, source, second, count[k] >> f;
        close(f);
    }
    
    # Write summary CSV: per server, per second
    for (ksum in client_sum) {
        split(ksum, q, "|");
        server = q[1]; second = q[2];
        c = client_sum[ksum] + 0;
        r = repl_sum[ksum] + 0;
        t = c + r;
        printf "%s,%s,%s,%d,%d,%d\n", exp_name, server, second, c, r, t >> summary_csv;
    }

    # Also write a per-server pivot: columns client and each master IP
    # Structure: experiment,server,second,elapsed_seconds,client,<master_ip1>,<master_ip2>,...,total
    # Master IPs are dynamically discovered from the logs - no hardcoding
    for (srv in servers) {
        # Build header: client first, then all master sources discovered from logs
        # Dynamically collect all unique master sources found in the logs (excluding "client")
        delete master_list;
        master_count = 0;
        for (src_key in sources) {
            if (src_key != "client") {
                master_count++;
                master_list[master_count] = src_key;
            }
        }
        # Sort master sources for consistent column order
        if (master_count > 0) {
            sort_array(master_list, master_count);
        }
        # Build header
        hdr = "experiment,server,second,elapsed_seconds,client";
        for (i = 1; i <= master_count; i++) {
            hdr = hdr "," master_list[i];
        }
        hdr = hdr ",total";
        pf = per_server_dir "/" srv "_breakdown.csv";
        print hdr > pf;
        
        # For each second that had any activity for this server
        for (ks in secs) {
            split(ks, xs, "|");
            if (xs[1] != srv) continue;
            second = xs[2];
            elapsed = second - minsec[srv];
            # Client count
            c = count[srv "|client|" second] + 0;
            total = c;
            line = exp_name "," srv "," second "," elapsed "," c;
            # Add counts for each master source
            for (i = 1; i <= master_count; i++) {
                src = master_list[i];
                v = count[srv "|" src "|" second] + 0;
                line = line "," v;
                total += v;
            }
            line = line "," total;
            print line >> pf;
        }
        close(pf);
    }
}
' "${log_files[@]}" >> "$out_csv"

echo "✅ Wrote: $out_csv"
echo "✅ Wrote: $summary_csv"
echo "✅ Wrote per-server breakdowns to: $per_server_dir/"

echo "Done."
