#!/bin/bash

# Analyze KeyDB CMD_LOG entries to compute per-server throughput broken down by source
# Usage: ./analyze_cmd_throughput.sh <server_log_dir>
# server_log_dir layout is expected as: <root>/experiment_*/mteverest*/mteverest*.log

# set -euo pipefail

# if [ $# -lt 1 ]; then
#     echo "Usage: $0 <server_log_dir>" 1>&2
#     exit 1
# fi
#  9
server_log_dir="/home/abhattar/keydb_logs/ember_data/mlp_sync/experiment_3"


timestamp_dir=$(date +%Y%m%d_%H%M%S)
output_dir="cmd_throughput_${timestamp_dir}"
mkdir -p "$output_dir"
echo "📁 Results will be saved to: $output_dir"

found_any=false

# Support two layouts:
# 1) <server_log_dir>/experiment_*/mteverest*/mteverest*.log
# 2) <server_log_dir>/mteverest*/mteverest*.log
for exp_dir in "$server_log_dir"/experiment_* "$server_log_dir"; do
    [ -d "$exp_dir" ] || continue
    found_any=true
    exp_name=$(basename "$exp_dir")

    # Collect mteverest*.log files
    log_files=()
    for server_dir in "$exp_dir"/mteverest*/; do
        [ -d "$server_dir" ] || continue
        for log_file in "$server_dir"/*.log; do
            [ -f "$log_file" ] && log_files+=("$log_file")
        done
    done

    if [ ${#log_files[@]} -eq 0 ]; then
        echo "⚠️  No mteverest*.log files found in $exp_name"
        continue
    fi

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
    index($0, "CMD_LOG,") {
        # Extract record after CMD_LOG,
        pos = index($0, "CMD_LOG,");
        rec = substr($0, pos + 8);
        # Fields: ts_ms, client_id, role, peer, command
        split(rec, arr, ",");
        if (length(arr) < 5) next;
        ts_ms = arr[1] + 0;
        role = arr[3];
        peer = arr[4];
        second = int(ts_ms / 1000);
        # Determine server from filename (basename without .log)
        fname = basename(FILENAME);
        server = without_ext(fname);
        # Source: client vs specific upstream hostname (peer)
        source = (role == "client") ? "client" : peer;
        key = server "|" source "|" second;
        count[key]++;
        # Aggregate into client vs replicated buckets
        ksum = server "|" second;
        if (role == "client") {
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

        # Also write a per-server pivot similar to log_analyze: columns client and each other server
        # Determine full set of servers observed
        for (srv in servers) {
            # Identify other servers (exclude client and self)
            # Build header dynamically: experiment,server,second,elapsed_seconds,client,<srvA>,<srvB>,total
            hdr = "experiment,server,second,elapsed_seconds,client";
            n = 0;
            split("", others);
            for (s2 in servers) {
                if (s2 != srv) {
                    n++;
                    others[n] = s2;
                    hdr = hdr "," s2;
                }
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
                c = count[srv "|client|" second] + 0;
                total = c;
                line = exp_name "," srv "," second "," elapsed "," c;
                for (i = 1; i <= n; i++) {
                    src = others[i];
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
done

if ! $found_any; then
    echo "⚠️  No experiment_* directories found under: $server_log_dir"
    exit 1
fi

echo "Done."


