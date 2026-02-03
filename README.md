# Distributed Machine Learning with script.sh

This project runs **distributed machine learning** across multiple servers using **KeyDB** (Redis-compatible) for coordination. The `script.sh` file automates setup, data sharding, weight initialization, training (MLP/CNN, Ember/CIC-IDS/UNSW), and log collection.

---

## Overview

- **Servers:** `mteverest1`, `mteverest3`, `mteverest4` (configurable at the top of `script.sh`)
- **Coordination:** KeyDB (multi-master Redis) on port 6379
- **Models:** MLP and CNN; **datasets:** Ember 2018, CIC-IDS2017, UNSW-NB15
- **Modes:** *Eventual consistency* (async) or *sequential* (sync) distributed training

---

## Prerequisites

1. **SSH access** to all servers (e.g. `username@server.uwyo.edu`).
2. **sshpass** installed (for non-interactive SSH with password):
   ```bash
   # Ubuntu/Debian
   sudo apt-get install sshpass
   ```
3. **Password for SSH** — set one of:
   - **Option A (recommended):** `export ML_CARS_PASSWORD=your_password`
   - **Option B:** Create `~/.ml_cars_password` with the password (first line), then `chmod 600 ~/.ml_cars_password`
   - **Option C:** `export PROMPT_PASSWORD=1` to be prompted when the script runs
   - **Option D:** Set `PASSWORD="..."` at the top of `script.sh` (avoid committing this)

4. **Python 3** and **venv** on each server; **KeyDB** built in `~/KeyDB` on each server (see setup steps below).

---

## How to Run script.sh

The script **does not have an interactive menu**. You run it in one of two ways:

### 1. Call a specific function (recommended)

Source the script, then call the function you need:

```bash
cd /home/abhattar/ml_cars_main
source script.sh
test_ssh                                    # Test SSH to all servers
copy_ml_cars_to_servers                     # Deploy code to servers
copy_and_build_keydb_on_servers             # Deploy and build KeyDB
# ... etc.
```

### 2. Run the default pipeline at the end of the script

The script ends by calling **`train_2`** (see bottom of `script.sh`). So a plain run:

```bash
./script.sh
```

executes the `train_2` workflow (CNN, eventual consistency, then copy logs). To run a different workflow, either:

- **Edit the end of `script.sh`** and replace `train_2` with another function (e.g. `train_1`), or  
- **Comment out** the final `train_2` and use **option 1** to call functions manually.

---

## Script Functions Reference

### Server and SSH

| Function | Description |
|----------|-------------|
| `test_ssh` | Test SSH to each server and show server map. |
| `generate_map` | (Internal) Build server/client maps. |
| `display_map` | Print server and client mappings. |

### Deploying code and KeyDB

| Function | Description |
|----------|-------------|
| `copy_ml_cars_to_servers` | Copy full `ml_cars` tree to each server (excludes KeyDB, venv, parquet). Requires `nodes.txt` in project root. |
| `copy_distributed_trainer_to_servers` | Copy only trainer and initializer Python files to `~/ml_cars` on each server. |
| `copy_and_build_keydb_on_servers` | Copy KeyDB source to each server and build (`make -j10`). |
| `copy_keydb_config` | Copy KeyDB config to servers. |
| `check_ml_cars_on_servers` | Verify `~/ml_cars` exists on all servers. |

### Python environment on servers

| Function | Description |
|----------|-------------|
| `create_python_envs_on_servers` | Create `python3 -m venv myenv` in `~/ml_cars` on each server. |
| `install_requirements_on_servers` | Run `pip install -r requirements.txt` in `~/ml_cars` venv on each server. |
| `install_system_packages_on_servers` | Install system packages if needed. |
| `check_sklearn_on_servers` | Check that scikit-learn is installed in venv on each server. |

### Dataset shards

Prepare shards **locally** first (see “Data preparation” below), then copy:

| Function | Description |
|----------|-------------|
| `copy_dataset_shards_to_servers` | Copy **Ember** train/test shards to `~/data/` (e.g. `train_shard.parquet`, `test_shard.parquet`). |
| `copy_unsw_dataset_shards_to_servers` | Copy **UNSW-NB15** shards to `~/unsw_data/` on each server. |
| `copy_cic_dataset_shards_to_servers` | Copy **CIC-IDS2017** shards to `~/cic_shard/` (e.g. `cic_train.parquet`, `cic_test.parquet`). |

### KeyDB lifecycle

| Function | Description |
|----------|-------------|
| `start_servers` | Start KeyDB on all servers (multi-master, active-replica, port 6379). |
| `close_servers` | Stop KeyDB on all servers. |
| `delete_database` | Remove RDB/AOF files in `~/KeyDB` on each server. |
| `delete_keydb_logs` | Delete KeyDB log files on servers. |

### Weight initialization

Run **once** before starting distributed training (so KeyDB has initial model weights):

| Function | Description |
|----------|-------------|
| `initialize_mlp_weights_on_yangra1` | Initialize **Ember MLP** weights (runs on mteverest4). |
| `initialize_cnn_weights_on_yangra1` | Initialize **Ember CNN** weights (runs on mteverest1). |
| `intitialize_cic_weights_on_servers` | Initialize **CIC-IDS MLP** weights (runs on mteverest1). |
| `initialize_cic_cnn_weights_on_servers` | Initialize **CIC-IDS CNN** weights (runs on mteverest1). |

### Starting distributed training

| Function | Description |
|----------|-------------|
| `start_distributed_training` | **Ember** – eventual consistency MLP (`distributed_mlp_trainer.py`). |
| `start_seq_distributed_training` | **Ember** – sequential MLP (`seq_distributed_mlp_trainer.py`). |
| `start_distributed_cnn_training` | **Ember** – eventual consistency CNN (`distributed_cnn_trainer.py`). |
| `start_seq_distributed_cnn_training` | **Ember** – sequential CNN (`seq_distributed_cnn_trainer.py`). |
| `start_distributed_cic_training` | **CIC-IDS** – eventual consistency MLP. |
| `start_seq_distributed_cic_training` | **CIC-IDS** – sequential MLP. |
| `start_distributed_cic_cnn_training` | **CIC-IDS** – eventual consistency CNN. |
| `start_seq_distributed_cic_cnn_training` | **CIC-IDS** – sequential CNN. |

CIC trainers expect env vars (set by the script on the server):  
`CIC_TRAIN_FILE=../cic_shard/cic_train.parquet`, `CIC_TEST_FILE=../cic_shard/cic_test.parquet`.

### Monitoring and stopping

| Function | Description |
|----------|-------------|
| `start_training_monitor` | Run local `training_monitor.py` (foreground). |
| `check_training_status` | Run `training_monitor.py --detailed`. |
| `stop_distributed_training` | Stop training using PID files (Ember). |
| `stop_all_distributed_ml_processes` | Kill all distributed trainer and weight-initializer processes on all servers. |

### Logs

| Function | Description |
|----------|-------------|
| `copy_keydb_logs` | Copy KeyDB logs from each server to local. |
| `copy_logs` | Copy training logs from each server to local. |
| `copy_training_logs_from_servers` | Copy training CSVs, `training_*.log`, and PID files into a local directory per server. |
| `copy_all_logs_to_mteverest4` | Gather all logs from all servers into a single directory on mteverest4. |

---

## Data preparation (before copying shards)

- **Ember:** Create shards with `shard_parquet.py`. Expected pattern:  
  `train_ember_2018_v2_features.cleaned.shard{0,1,2}.parquet` and same for `test_...`.  
  Then run `copy_dataset_shards_to_servers`.
- **UNSW-NB15:** Create shards (e.g. with `unsw_shard_parquet.py`) under `unsw_shard/`  
  (e.g. `train_unsw_nb15.shard0.parquet`). Then run `copy_unsw_dataset_shards_to_servers`.
- **CIC-IDS2017:** Create shards under `cic_shard/` (e.g. `cic_train.shard0.parquet`, `cic_test.shard0.parquet`).  
  Then run `copy_cic_dataset_shards_to_servers`.

Server-to-shard mapping in the script: mteverest1→0, mteverest3→1, mteverest4→2.

---

## Example workflows

### One-time setup (first time)

```bash
source script.sh
test_ssh
copy_ml_cars_to_servers
copy_and_build_keydb_on_servers
create_python_envs_on_servers
install_requirements_on_servers
# Then copy the dataset shards you need, e.g.:
copy_dataset_shards_to_servers
# or copy_cic_dataset_shards_to_servers
# or copy_unsw_dataset_shards_to_servers
```

### Ember – sequential CNN training (full run)

```bash
source script.sh
close_servers
delete_database
delete_keydb_logs
copy_distributed_trainer_to_servers
stop_all_distributed_ml_processes
start_servers
sleep 10
initialize_cnn_weights_on_yangra1
sleep 10
start_seq_distributed_cnn_training
# Let it run; then:
# close_servers
# copy_keydb_logs
# copy_logs
# copy_training_logs_from_servers
```

### Ember – eventual consistency CNN training

Same as above, but call `start_distributed_cnn_training` instead of `start_seq_distributed_cnn_training`.

### CIC-IDS – MLP or CNN

After copying CIC shards and deploying code:

```bash
copy_cic_dataset_shards_to_servers
# For MLP:
intitialize_cic_weights_on_servers
start_distributed_cic_training   # or start_seq_distributed_cic_training
# For CNN:
initialize_cic_cnn_weights_on_servers
start_distributed_cic_cnn_training   # or start_seq_distributed_cic_cnn_training
```

### Using the built-in pipelines

- **`train_1`** (in script): Sequential CNN on Ember — close KeyDB, delete DB/logs, copy trainers, stop old ML processes, start KeyDB, init CNN weights, run `start_seq_distributed_cnn_training`, wait 150m, then close servers and copy logs.
- **`train_2`** (default at end of script): Same idea but runs `start_distributed_cnn_training` (eventual consistency) and waits 1600 seconds.

Edit the last line(s) of `script.sh` to call `train_1`, `train_2`, or your own sequence of the functions above.

---

## Configuration

- **Servers:** Edit the `servers` array at the top of `script.sh`:
  ```bash
  servers=("mteverest1" "mteverest3" "mteverest4")
  ```
- **Username:** Set `username="abhattar"` (or your SSH user).
- **nodes.txt:** Required by `copy_ml_cars_to_servers`; format per line:  
  `user@host.uwyo.edu password`

---

## Summary

| Goal | What to run |
|------|-------------|
| Test SSH | `source script.sh; test_ssh` |
| Deploy code + KeyDB + env | Setup workflow above |
| Copy Ember/CIC/UNSW shards | `copy_*_shards_to_servers` for your dataset |
| Start KeyDB | `start_servers` |
| Init weights (Ember MLP/CNN) | `initialize_mlp_weights_on_yangra1` or `initialize_cnn_weights_on_yangra1` |
| Init weights (CIC) | `intitialize_cic_weights_on_servers` or `initialize_cic_cnn_weights_on_servers` |
| Run training | One of `start_distributed_*` or `start_seq_distributed_*` |
| Monitor | `check_training_status` or `start_training_monitor` |
| Stop training | `stop_all_distributed_ml_processes` or `stop_distributed_training` |
| Get logs | `copy_logs`, `copy_keydb_logs`, `copy_training_logs_from_servers` |

For a single command that runs a full training pipeline, set the last line of `script.sh` to the function you want (e.g. `train_1` or `train_2`) and run `./script.sh`.
