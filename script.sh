servers=("mteverest1" "mteverest3" "mteverest4")


declare -A machine_server_map
declare -A replicas_map 


username="abhattar"

# -----------------------------------------------------------------------------
# Password configuration
# You can configure the SSH password in one of the following (preferred order):
# 1) Set PASSWORD at the top of this script (not recommended to commit to VCS).
#    Example: PASSWORD="mysecret"
# 2) Export ML_CARS_PASSWORD in your environment: export ML_CARS_PASSWORD=...
# 3) Create a file at $HOME/.ml_cars_password containing the password (chmod 600).
# 4) Set PROMPT_PASSWORD=1 to prompt interactively on script run (read -s).
# If none of the above are present, the script will fall back to the historical
# default password to avoid breaking existing workflows, but a warning will be
# printed. Prefer using environment or file options to keep secrets out of git.
# -----------------------------------------------------------------------------

# Optionally set PASSWORD here (override with env var or prompt as needed):
PASSWORD="${PASSWORD:-}"



get_password() {
    # Resolve the password from multiple sources (in this order)
    if [ -n "$PASSWORD" ]; then
        password="$PASSWORD"
    elif [ -n "${ML_CARS_PASSWORD:-}" ]; then
        password="$ML_CARS_PASSWORD"
    elif [ -f "$HOME/.ml_cars_password" ]; then
        # read first line from the secret file
        password="$(sed -n '1p' "$HOME/.ml_cars_password")"
    elif [ "${PROMPT_PASSWORD:-}" = "1" ]; then
        # prompt the user (no echo)
        read -s -p "Enter SSH password: " password
        echo
    else
        echo "Warning: No SSH password provided (env/file/PASSWORD/prompts). Falling back to default (insecure)." >&2
        echo "To avoid this, set ML_CARS_PASSWORD, create ~/.ml_cars_password (chmod 600), or set PASSWORD at the top of this script." >&2
        password="$DEFAULT_PASSWORD"
    fi
}

# Initialize global password variable used throughout the script
get_password
generate_map(){
    server_node_id_counter=0
    for server in "${servers[@]}"; do
        machine_server_map["$server"]="${username}@$server.uwyo.edu"
    done

    client_node_id_counter=0
    for client in "${clients[@]}"; do
        machine_client_map["$client"]="${username}@$client.uwyo.edu"
    done

    for server in "${servers[@]}"; do
        replicas_map["$server"]=""
        for replica in "${servers[@]}"; do
            if [[ "$server" != "$replica" ]]; then
                replicas_map["$server"]+="$replica "
            fi
        done
    done

    
}


display_map(){
    for key in "${!machine_server_map[@]}"; do
        echo " server $key : ${machine_server_map[$key]}"
    done

    for key in "${!machine_client_map[@]}"; do
        echo " client $key : ${machine_client_map[$key]}"
    done
}



test_ssh(){
        generate_map
        display_map
        echo "Attempting to SSH into each server and checking if redis is running on port 6379..."
        for key in "${!machine_server_map[@]}"; do
            echo "Connecting to ${machine_server_map[$key]}..."
            ssh "${machine_server_map[$key]}" "pwd"
        done

      


    }

copy_distributed_trainer_to_servers() {
    generate_map
    
    echo "--------------------------------------------------------------------------"
    echo "                    Copying distributed_mlp_trainer.py to Servers            "    
    echo "--------------------------------------------------------------------------"
    
    # Check if distributed_mlp_trainer.py exists
    if [ ! -f "distributed_mlp_trainer.py" ]; then
        echo "❌ Error: distributed_mlp_trainer.py not found in current directory!"
        return 1
    fi
    
    # Ensure password is set (preserve any password resolved earlier)
    password="${password:-$DEFAULT_PASSWORD}"
    
    for key in "${!machine_server_map[@]}"
    do
        destination=$key
        echo "Copying distributed_mlp_trainer.py to server: $destination"
        
        # Delete logs directory on remote server first
        echo "  Deleting logs directory on remote server..."
        sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@$destination "rm -rf ~/ml_cars/logs"
        
        # Copy the file directly using scp
        echo "  Copying file to $username@$destination..."
        sshpass -p "$password" scp -o StrictHostKeyChecking=no distributed_mlp_trainer.py $username@$destination:~/ml_cars/
        sshpass -p "$password" scp -o StrictHostKeyChecking=no seq_distributed_mlp_trainer.py $username@$destination:~/ml_cars/
        
        sshpass -p "$password" scp -o StrictHostKeyChecking=no distributed_cnn_trainer.py $username@$destination:~/ml_cars/
        sshpass -p "$password" scp -o StrictHostKeyChecking=no cnn_weights_initializer.py $username@$destination:~/ml_cars/
        sshpass -p "$password" scp -o StrictHostKeyChecking=no lstm_weights_initializer.py $username@$destination:~/ml_cars/
        sshpass -p "$password" scp -o StrictHostKeyChecking=no distributed_lstm_trainer.py $username@$destination:~/ml_cars/
        sshpass -p "$password" scp -o StrictHostKeyChecking=no seq_distributed_cnn_trainer.py $username@$destination:~/ml_cars/
        sshpass -p "$password" scp -o StrictHostKeyChecking=no weight_intializer.py $username@$destination:~/ml_cars/
        
        # Copy UNSW-NB15 training and initializing files
        sshpass -p "$password" scp -o StrictHostKeyChecking=no distributed_unsw_mlp_trainer.py $username@$destination:~/ml_cars/
        sshpass -p "$password" scp -o StrictHostKeyChecking=no unsw_weight_initializer.py $username@$destination:~/ml_cars/

        # Copy CIC-IDS2017 training and initializer files
        sshpass -p "$password" scp -o StrictHostKeyChecking=no distributed_cic_mlp_trainer.py $username@$destination:~/ml_cars/
        sshpass -p "$password" scp -o StrictHostKeyChecking=no seq_distributed_cic_mlp_trainer.py $username@$destination:~/ml_cars/
        sshpass -p "$password" scp -o StrictHostKeyChecking=no cic_weight_initializer.py $username@$destination:~/ml_cars/
        # Copy CIC-IDS2017 CNN training and initializer files
        sshpass -p "$password" scp -o StrictHostKeyChecking=no distributed_cic_cnn_trainer.py $username@$destination:~/ml_cars/
        sshpass -p "$password" scp -o StrictHostKeyChecking=no seq_distributed_cic_cnn_trainer.py $username@$destination:~/ml_cars/
        sshpass -p "$password" scp -o StrictHostKeyChecking=no cic_cnn_weights_initializer.py $username@$destination:~/ml_cars/
        
        # Verify the copy
        echo "  Verifying file on remote server..."
        sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@$destination "ls -la ~/ml_cars/distributed_mlp_trainer.py"
        
        if [ $? -eq 0 ]; then
            echo "  ✅ Successfully copied to $destination"
        else
            echo "  ❌ Failed to copy to $destination"
        fi
        
        echo
    done
    
    echo "✅ distributed_mlp_trainer.py copied to all servers successfully!"
    echo "🗑️  Old logs directories deleted from all servers"
}



copy_ml_cars_to_servers() {
    generate_map
    
    echo "--------------------------------------------------------------------------"
    echo "                    Copying ml_cars to Servers                              "    
    echo "--------------------------------------------------------------------------"
    
    # Check if nodes.txt exists
    if [ ! -f "nodes.txt" ]; then
        echo "Error: nodes.txt not found in current directory!"
        echo "Please create nodes.txt with format:"
        echo "abhattar@yangra3.uwyo.edu password3"
        echo "abhattar@yangra6.uwyo.edu password6"
        echo "abhattar@yangra1.uwyo.edu password1"
        return 1
    fi
    
    # Create a compressed tar of ml_cars (excluding KeyDB and parquet files)
    echo "Creating compressed archive of ml_cars (excluding KeyDB and parquet files)..."
    cd /home/abhattar
    tar -czf ml_cars_no_keydb.tar.gz \
        --exclude='ml_cars_main/KeyDB' \
         --exclude='ml_cars_main/myenv' \
        --exclude='ml_cars_main/*.parquet' \
        --exclude='ml_cars_main/train_ember_2018_v2_features.parquet' \
        --exclude='ml_cars_main/test_ember_2018_v2_features.parquet' \
        --exclude='ml_cars_main/training_logs_*' \
        ml_cars_main/
    
    for key in "${!machine_server_map[@]}"
    do
        destination=$key
        target_directory="ml_cars"
        
        echo "Copying to server: $destination"
        echo "  Target directory: $target_directory"
        
    # Get password for this server from nodes.txt
    # If you want per-server passwords, implement parsing here and set
    # `password` accordingly. By default we use the centrally resolved value.
    password="${password:-$DEFAULT_PASSWORD}"
        
        if [ -z "$password" ]; then
            echo "  Error: No password found for $username@$destination.uwyo.edu in nodes.txt"
            continue
        fi
        
        # Create directory and clean it on remote server
        echo "  Creating directory on remote server..."
        sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@$destination "mkdir -p $target_directory; cd $target_directory; rm -rf *"
        
        # Copy the compressed file using scp instead of rsync
        echo "  Copying compressed archive to $username@$destination..."
        sshpass -p "$password" scp -o StrictHostKeyChecking=no ml_cars_no_keydb.tar.gz $username@$destination:~/$target_directory/
        
        # Extract and set up on remote server
        echo "  Extracting files on remote server..."
        sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@$destination "cd $target_directory; tar -xzf ml_cars_no_keydb.tar.gz --strip-components=1; rm ml_cars_no_keydb.tar.gz"
        
        echo "  Done with $destination"
        echo
    done
    
    # Clean up local tar file
    rm -f ml_cars_no_keydb.tar.gz
    cd ~/ml_cars
    
    echo "ml_cars directory copied to all servers successfully!"
    echo "Note: Parquet files were excluded to reduce transfer size"
}



copy_and_build_keydb_on_servers() {
    generate_map
    
    echo "--------------------------------------------------------------------------"
    echo "                    Copying and Building KeyDB on Servers                  "    
    echo "--------------------------------------------------------------------------"
    
    # Check if nodes.txt exists
    if [ ! -f "nodes.txt" ]; then
        echo "Error: nodes.txt not found in current directory!"
        return 1
    fi
    
    # Create a compressed tar of just the KeyDB folder from ml_cars
    echo "Creating compressed archive of KeyDB from ml_cars..."
    cd /home/abhattar/ml_cars_main
    tar -czf KeyDB.tar.gz KeyDB/
    
    for key in "${!machine_server_map[@]}"
    do
        destination=$key
        target_directory="KeyDB"
        
        echo "Setting up KeyDB on server: $destination"
        echo "  Target directory: $target_directory"
        
    # Get password for this server from nodes.txt
    # If you want per-server passwords, implement parsing here and set
    # `password` accordingly. By default we use the centrally resolved value.
    password="${password:-$DEFAULT_PASSWORD}"
        
        if [ -z "$password" ]; then
            echo "  Error: No password found for $username@$destination.uwyo.edu in nodes.txt"
            continue
        fi
        
        # Create directory and clean it on remote server
        echo "  Creating directory on remote server..."
        sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@$destination "mkdir -p $target_directory; cd $target_directory; rm -rf *"
        
        # Copy the compressed KeyDB folder using scp
        echo "  Copying KeyDB archive to $username@$destination..."
        sshpass -p "$password" scp -o StrictHostKeyChecking=no KeyDB.tar.gz $username@$destination:~/$target_directory/
        
        # Extract and build on remote server
        echo "  Extracting and building KeyDB on remote server..."
        sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@$destination "cd $target_directory; tar -xzf KeyDB.tar.gz --strip-components=1; rm KeyDB.tar.gz; make clean; make -j10"
        
        echo "  Done with $destination"
        echo
    done
    
    # Clean up local tar file
    rm -f KeyDB.tar.gz
    cd ~/ml_cars
    
    echo "KeyDB copied and built on all servers successfully!"
}


check_ml_cars_on_servers() {
    generate_map
    
    echo "--------------------------------------------------------------------------"
    echo "                    Checking ml_cars on Servers                            "    
    echo "--------------------------------------------------------------------------"
    
    password="${password:-$DEFAULT_PASSWORD}"
    
    for key in "${!machine_server_map[@]}"
    do
        destination=$key
        target_directory="ml_cars"
        
        echo "Checking server: $destination"
        echo "  Target directory: $target_directory"
        
        # Check if directory exists and list contents
        echo "  Checking if directory exists..."
        if sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@$destination "test -d ~/$target_directory"; then
            echo "  ✅ Directory exists on $destination"
            
            # Count files and show directory size
            file_count=$(sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@$destination "find ~/$target_directory -type f | wc -l")
            dir_size=$(sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@$destination "du -sh ~/$target_directory | cut -f1")
            echo "  📁 Directory size: $dir_size, Files: $file_count"
            
            # List all files in the directory
            echo "  📋 Files in $target_directory:"
            sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@$destination "ls -la ~/$target_directory" | while read line; do
                echo "    $line"
            done
            
            # Check for specific important files
            echo "  🔍 Checking for key files:"
            if sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@$destination "test -f ~/$target_directory/mlp_trainer.py"; then
                echo "    ✅ mlp_trainer.py found"
            else
                echo "    ❌ mlp_trainer.py missing"
            fi
            
            if sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@$destination "test -f ~/$target_directory/weight_intializer.py"; then
                echo "    ✅ weight_intializer.py found"
            else
                echo "    ❌ weight_intializer.py missing"
            fi
            
            if sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@$destination "test -f ~/$target_directory/preprocess_to_parquet.py"; then
                echo "    ✅ preprocess_to_parquet.py found"
            else
                echo "    ❌ preprocess_to_parquet.py missing"
            fi
            
        else
            echo "  ❌ Directory does NOT exist on $destination"
        fi
        
        echo
    done
    
    echo "ml_cars directory check completed!"
}
copy_dataset_shards_to_servers() {
    generate_map
    
    echo "--------------------------------------------------------------------------"
    echo "                    Copying Dataset Shards to Servers                      "    
    echo "--------------------------------------------------------------------------"
    
    # Check if shard files exist
    if [ ! -f "/home/abhattar/ml_cars_main/train_ember_2018_v2_features.cleaned.shard0.parquet" ]; then
        echo "Error: Dataset shards not found! Please run shard_parquet.py first."
        return 1
    fi
    
    password="${password:-$DEFAULT_PASSWORD}"
    
    # Map servers to shard indices
    declare -A server_to_shard
    server_to_shard["mteverest1"]=0
    server_to_shard["mteverest3"]=1
    server_to_shard["mteverest4"]=2
    
    for key in "${!machine_server_map[@]}"
    do
        destination=$key
        shard_index=${server_to_shard[$destination]}
        
        echo "Copying shard $shard_index to server: $destination"
        
        # Create data directory on remote server
        echo "  Creating data directory..."
        sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@$destination "mkdir -p ~/data"
        
        # Copy train shard
        echo "  Copying train shard $shard_index..."
        local_train_shard="/home/abhattar/ml_cars_main/train_ember_2018_v2_features.cleaned.shard${shard_index}.parquet"
        sshpass -p "$password" scp -o StrictHostKeyChecking=no "$local_train_shard" $username@$destination:~/data/train_shard.parquet
        
        # Copy test shard
        echo "  Copying test shard $shard_index..."
        local_test_shard="/home/abhattar/ml_cars_main/test_ember_2018_v2_features.cleaned.shard${shard_index}.parquet"
        sshpass -p "$password" scp -o StrictHostKeyChecking=no "$local_test_shard" $username@$destination:~/data/test_shard.parquet
        
        # Verify the copy
        echo "  Verifying files on remote server..."
        sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@$destination "ls -lh ~/data/"
        
        echo "  Done with $destination (shard $shard_index)"
        echo
    done
    
    echo "Dataset shards copied to all servers successfully!"
    echo "Each server now has its corresponding train and test shard."
}

copy_unsw_dataset_shards_to_servers() {
    generate_map
    
    echo "--------------------------------------------------------------------------"
    echo "                    Copying UNSW Dataset Shards to Servers                      "    
    echo "--------------------------------------------------------------------------"
    
    # Check if UNSW shard files exist
    if [ ! -f "/home/abhattar/ml_cars_main/unsw_shard/train_unsw_nb15.shard0.parquet" ]; then
        echo "Error: UNSW dataset shards not found! Please run unsw_shard_parquet.py first."
        return 1
    fi
    
    password="${password:-$DEFAULT_PASSWORD}"
    
    # Map servers to shard indices
    declare -A server_to_shard
    server_to_shard["mteverest1"]=0
    server_to_shard["mteverest3"]=1
    server_to_shard["mteverest4"]=2
    
    for key in "${!machine_server_map[@]}"
    do
        destination=$key
        shard_index=${server_to_shard[$destination]}
        
        echo "Copying UNSW shard $shard_index to server: $destination"
        
        # Create unsw_data directory on remote server
        echo "  Creating unsw_data directory..."
        sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@$destination "mkdir -p ~/unsw_data"
        
        # Copy train shard
        echo "  Copying UNSW train shard $shard_index..."
        local_train_shard="/home/abhattar/ml_cars_main/unsw_shard/train_unsw_nb15.shard${shard_index}.parquet"
        sshpass -p "$password" scp -o StrictHostKeyChecking=no "$local_train_shard" $username@$destination:~/unsw_data/train_shard.parquet
        
        # Copy test shard
        echo "  Copying UNSW test shard $shard_index..."
        local_test_shard="/home/abhattar/ml_cars_main/unsw_shard/test_unsw_nb15.shard${shard_index}.parquet"
        sshpass -p "$password" scp -o StrictHostKeyChecking=no "$local_test_shard" $username@$destination:~/unsw_data/test_shard.parquet
        
        # Verify the copy
        echo "  Verifying files on remote server..."
        sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@$destination "ls -lh ~/unsw_data/"
        
        echo "  Done with $destination (UNSW shard $shard_index)"
        echo
    done
    
    echo "UNSW dataset shards copied to all servers successfully!"
    echo "Each server now has its corresponding UNSW train and test shard in ~/unsw_data/"
}

# Copy CIC-IDS2017 dataset shards to servers (modeled after UNSW function)
copy_cic_dataset_shards_to_servers() {
    generate_map
    
    echo "--------------------------------------------------------------------------"
    echo "                    Copying CIC-IDS Dataset Shards to Servers                 "    
    echo "--------------------------------------------------------------------------"
    
    # Check if CIC shard files exist
    if [ ! -f "/home/abhattar/ml_cars_main/cic_shard/cic_train.shard0.parquet" ]; then
        echo "Error: CIC-IDS dataset shards not found! Please run cic_shard_parquet.py first."
        return 1
    fi
    
    password="${password:-$DEFAULT_PASSWORD}"
    
    # Map servers to shard indices
    declare -A server_to_shard
    server_to_shard["mteverest1"]=0
    server_to_shard["mteverest3"]=1
    server_to_shard["mteverest4"]=2
    
    for key in "${!machine_server_map[@]}"
    do
        destination=$key
        shard_index=${server_to_shard[$destination]}
        
        echo "Copying CIC shard $shard_index to server: $destination"
        
        # Create cic_shard directory on remote server
        echo "  Creating cic_shard directory..."
        sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@$destination "mkdir -p ~/cic_shard"
        
        # Copy train shard
        echo "  Copying CIC train shard $shard_index..."
        local_train_shard="/home/abhattar/ml_cars_main/cic_shard/cic_train.shard${shard_index}.parquet"
        sshpass -p "$password" scp -o StrictHostKeyChecking=no "$local_train_shard" $username@$destination:~/cic_shard/cic_train.parquet
        
        # Copy test shard
        echo "  Copying CIC test shard $shard_index..."
        local_test_shard="/home/abhattar/ml_cars_main/cic_shard/cic_test.shard${shard_index}.parquet"
        sshpass -p "$password" scp -o StrictHostKeyChecking=no "$local_test_shard" $username@$destination:~/cic_shard/cic_test.parquet
        
        # Verify the copy
        echo "  Verifying files on remote server..."
        sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@$destination "ls -lh ~/cic_shard/"
        
        echo "  Done with $destination (CIC shard $shard_index)"
        echo
    done
    
    echo "CIC-IDS dataset shards copied to all servers successfully!"
    echo "Each server now has its corresponding CIC train and test shard in ~/cic_shard/"
}

# Add this function to your script.sh and call it:
# copy_dataset_shards_to_servers
create_python_envs_on_servers() {
    generate_map
    
    echo "--------------------------------------------------------------------------"
    echo "                    Creating Python Environments on Servers                  "    
    echo "--------------------------------------------------------------------------"
    
    password="${password:-$DEFAULT_PASSWORD}"
    
    for key in "${!machine_server_map[@]}"
    do
    destination=$key
    password="${password:-$DEFAULT_PASSWORD}"
        echo "Setting up Python environment on server: $destination"
        
        # Check if Python3 and venv are available
        echo "  Checking Python installation..."
        sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@$destination "python3 --version && python3 -m venv --help"
        
        # Create virtual environment in ml_cars directory
        echo "  Creating virtual environment..."
        sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@$destination "cd ~/ml_cars && python3 -m venv myenv"
        
        # Activate and upgrade pip
        echo "  Upgrading pip..."
        sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@$destination "cd ~/ml_cars && bash -c 'source myenv/bin/activate && pip install --upgrade pip'"
        
        echo "  Done with $destination"
        echo
    done
    
    echo "Python environments created on all servers successfully!"
}

install_requirements_on_servers() {
    generate_map
    
    echo "--------------------------------------------------------------------------"
    echo "                    Installing Requirements on Servers                      "    
    echo "--------------------------------------------------------------------------"
    
    password="${password:-$DEFAULT_PASSWORD}"
    
    for key in "${!machine_server_map[@]}"
    do
        destination=$key
        
        echo "Installing requirements on server: $destination"
        
        # Check if virtual environment exists
        echo "  Checking virtual environment..."
        if sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@$destination "test -d ~/ml_cars/myenv"; then
            echo "  ✅ Virtual environment found"
            
            # Install requirements from the existing requirements.txt
            echo "  Installing packages from requirements.txt..."
            sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@$destination "cd ~/ml_cars && bash -c 'source myenv/bin/activate && pip install -r requirements.txt'"
            
            # Verify installation
            echo "  Verifying key packages..."
            sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@$destination "cd ~/ml_cars && bash -c 'source myenv/bin/activate && python -c \"import torch, numpy, pandas, redis; print(\\\"All key packages imported successfully\\\")\"'"
            
        else
            echo "  ❌ Virtual environment not found. Please run create_python_envs_on_servers first."
            continue
        fi
        
        echo "  Done with $destination"
        echo
    done
    
    echo "Requirements installation completed on all servers!"
}
delete_keydb_logs() {
    generate_map
    
    echo "--------------------------------------------------------------------------"
    echo "                       Deleting KeyDB Logs on Servers                     "    
    echo "--------------------------------------------------------------------------"
    
    password="${password:-$DEFAULT_PASSWORD}"
    
    for key in "${!machine_server_map[@]}"
    do
        destination=$key
        echo "Deleting KeyDB logs on server: $destination"
        
        # Check and delete KeyDB log files
        sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@$destination "
            echo '  🔍 Checking KeyDB directory: ~/KeyDB'
            if [ -d ~/KeyDB ]; then
                cd ~/KeyDB
                echo '  📁 KeyDB directory found'
                echo '  📋 Current KeyDB files:'
                ls -la *.log *.rdb 2>/dev/null || echo '    No log/rdb files found'
                echo '  🗑️  Deleting KeyDB log files...'
                rm -f keydb_log.log dump.rdb *.log *.rdb
                echo '  ✅ KeyDB logs deleted from ~/KeyDB directory'
            else
                echo '  ❌ KeyDB directory not found at ~/KeyDB'
            fi
            
            # Also check ml_cars/KeyDB directory
            echo '  🔍 Checking KeyDB directory: ~/ml_cars/KeyDB'
            if [ -d ~/ml_cars/KeyDB ]; then
                cd ~/ml_cars/KeyDB
                echo '  📁 ml_cars/KeyDB directory found'
                echo '  📋 Current KeyDB files:'
                ls -la *.log *.rdb 2>/dev/null || echo '    No log/rdb files found'
                echo '  🗑️  Deleting KeyDB log files...'
                rm -f keydb_log.log dump.rdb *.log *.rdb
                echo '  ✅ KeyDB logs deleted from ~/ml_cars/KeyDB directory'
            else
                echo '  ❌ KeyDB directory not found at ~/ml_cars/KeyDB'
            fi
        "
        
        echo "  Done with $destination"
        echo
    done
    
    echo "KeyDB logs deletion completed on all servers!"
}

check_sklearn_on_servers() {
    generate_map
    
    echo "--------------------------------------------------------------------------"
    echo "                       Checking sklearn Installation on Servers           "    
    echo "--------------------------------------------------------------------------"
    
    password="${password:-$DEFAULT_PASSWORD}"
    
    for key in "${!machine_server_map[@]}"
    do
        destination=$key
        echo "Checking sklearn on server: $destination"
        
        # Check if sklearn is installed
        sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@$destination "bash -c '
            cd ~/ml_cars
            source myenv/bin/activate
            echo \"  🔍 Checking sklearn installation...\"
            python -c \"import sklearn; print(f\\\"✅ sklearn version: {sklearn.__version__}\\\")\" 2>/dev/null || echo \"❌ sklearn not installed\"
        '"
        
        echo "  Done with $destination"
        echo
    done
    
    echo "sklearn installation check completed on all servers!"
}

copy_keydb_logs() {
    generate_map
    display_map
    echo
    echo "---------------------------------------------------------------------------------"
    echo "$(date) Copying KeyDB Logs into home Keydb logs directory"
    echo "-----------------------------------------------------------------------------------"
    echo

    run_timestamp=$(date +"%Y%m%d_%H%M%S")
    local_base_directory="$HOME/Keydb logs"
    local_save_directory="$local_base_directory/$run_timestamp"

    mkdir -p "$local_save_directory"  # Ensure local directory exists (run-scoped)

    password="${password:-$DEFAULT_PASSWORD}"

    for key in "${!machine_server_map[@]}"
    do
        destination=$key
        echo "Copying KeyDB logs from $username@$destination..."

        local_time_start=$(date +%s)  # Start time tracking

        # Create per-server subdirectory to avoid overwrites across servers
        mkdir -p "$local_save_directory/$destination"

        # Copy KeyDB logs from ~/KeyDB directory
        sshpass -p "$password" scp -o StrictHostKeyChecking=no "$username@$destination:~/KeyDB/*.log" "$local_save_directory/$destination/" 2>/dev/null || echo "    No KeyDB logs found in ~/KeyDB"

        # Copy KeyDB RDB files
        sshpass -p "$password" scp -o StrictHostKeyChecking=no "$username@$destination:~/KeyDB/*.rdb" "$local_save_directory/$destination/" 2>/dev/null || echo "    No KeyDB RDB files found"

        local_time_end=$(date +%s)  # End time tracking
        local_time_duration=$(( local_time_end - local_time_start ))

        echo "        ... done in $local_time_duration seconds"
        echo
    done

    echo "✅ All KeyDB logs copied to: $local_save_directory"
    echo "📁 Check the '$local_base_directory' directory for accumulated KeyDB logs"
}
copy_logs() {
    generate_map
    display_map

    echo
    echo "---------------------------------------------------------------------------------"
    echo "$(date) Copying Logs into home serverlogs directory"
    echo "-----------------------------------------------------------------------------------"
    echo

    target_directory="~/ml_cars/logs"
    local_base_directory="$HOME/serverlogs"
    run_timestamp=$(date +"%Y%m%d_%H%M%S")
    local_save_directory="$local_base_directory/$run_timestamp"

    mkdir -p "$local_save_directory"  # Ensure run-scoped directory exists

    password="${password:-$DEFAULT_PASSWORD}"

    for key in "${!machine_server_map[@]}"
    do
        destination=$key
        echo "Copying logs from $username@$destination..."

        local_time_start=$(date +%s)  # Start time tracking

        # Create per-server subdirectory to avoid overwrites across runs/servers
        mkdir -p "$local_save_directory/$destination"

        # Copy CSV training logs from ml_cars/logs directory
        sshpass -p "$password" scp -o StrictHostKeyChecking=no \
            "$username@$destination:$target_directory/*.csv" "$local_save_directory/$destination/" \
            2>/dev/null || echo "    No CSV training logs found"

        local_time_end=$(date +%s)  # End time tracking
        local_time_duration=$(( local_time_end - local_time_start ))

        echo "        ... done in $local_time_duration seconds"
        echo
    done

    echo "✅ All logs copied to: $local_save_directory"
    echo "📁 Check the '$local_base_directory' directory for accumulated logs"
}


copy_all_logs_to_mteverest4() {
    generate_map
    
    echo "--------------------------------------------------------------------------"
    echo "                    Copying All Logs to mteverest4 Centralized Folder      "    
    echo "--------------------------------------------------------------------------"
    
    # Create centralized logs directory on mteverest4
    central_logs_dir="centralized_logs_$(date +%Y%m%d_%H%M%S)"
    echo "Creating centralized logs directory: $central_logs_dir"
    
    password="${password:-$DEFAULT_PASSWORD}"
    
    # Create the centralized directory on mteverest4
    sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@mteverest4 "mkdir -p ~/ml_cars_main/$central_logs_dir"
    
    for key in "${!machine_server_map[@]}"
    do
        destination=$key
        echo "Copying logs from server: $destination"
        
        # Create server-specific subdirectory on mteverest4
        sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@mteverest4 "mkdir -p ~/ml_cars_main/$central_logs_dir/$destination"
        
        # Copy training logs from ml_cars/logs directory
        echo "  📊 Copying training logs..."
        sshpass -p "$password" scp -o StrictHostKeyChecking=no -r "$username@$destination:~/ml_cars/logs/*" "$username@mteverest4:~/ml_cars_main/$central_logs_dir/$destination/" 2>/dev/null || echo "    No training logs found"
        
        # Copy training output logs (all training_*.log files)
        echo "  📝 Copying training output logs..."
        sshpass -p "$password" scp -o StrictHostKeyChecking=no "$username@$destination:~/ml_cars/training_*.log" "$username@mteverest4:~/ml_cars_main/$central_logs_dir/$destination/" 2>/dev/null || echo "    No training output logs found"
        
        # Copy CIC-specific training logs
        echo "  🔬 Copying CIC training logs..."
        sshpass -p "$password" scp -o StrictHostKeyChecking=no "$username@$destination:~/ml_cars/training_cic_*.log" "$username@mteverest4:~/ml_cars_main/$central_logs_dir/$destination/" 2>/dev/null || echo "    No CIC training logs found"
        
        # Copy KeyDB logs from both possible locations
        echo "  🗄️  Copying KeyDB logs..."
        sshpass -p "$password" scp -o StrictHostKeyChecking=no "$username@$destination:~/KeyDB/*.log" "$username@mteverest4:~/ml_cars_main/$central_logs_dir/$destination/keydb_logs/" 2>/dev/null || echo "    No KeyDB logs found in ~/KeyDB"
        sshpass -p "$password" scp -o StrictHostKeyChecking=no "$username@$destination:~/ml_cars/KeyDB/*.log" "$username@mteverest4:~/ml_cars_main/$central_logs_dir/$destination/keydb_logs/" 2>/dev/null || echo "    No KeyDB logs found in ~/ml_cars/KeyDB"
        
        # Copy PID files for status checking
        echo "  🔢 Copying PID files..."
        sshpass -p "$password" scp -o StrictHostKeyChecking=no "$username@$destination:~/ml_cars/*.pid" "$username@mteverest4:~/ml_cars_main/$central_logs_dir/$destination/" 2>/dev/null || echo "    No PID files found"
        
        # List what was copied
        echo "  📋 Files copied to mteverest4:"
        sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@mteverest4 "ls -la ~/ml_cars_main/$central_logs_dir/$destination/" 2>/dev/null || echo "    No files copied"
        
        echo "  Done with $destination"
        echo
    done
    
    # Create a summary file
    echo "  📄 Creating summary file..."
    sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@mteverest4 "
        cd ~/ml_cars_main/$central_logs_dir
        echo 'Centralized Logs Summary - $(date)' > summary.txt
        echo '=====================================' >> summary.txt
        echo '' >> summary.txt
        for server in */; do
            echo \"Server: \$server\" >> summary.txt
            echo \"Files:\" >> summary.txt
            ls -la \"\$server\" >> summary.txt
            echo '' >> summary.txt
        done
    "
    
    echo "✅ All logs copied to mteverest4: ~/ml_cars_main/$central_logs_dir"
    echo "📁 Each server has its own subdirectory with all logs"
    echo "📄 Summary file created: summary.txt"
    echo "📊 You can now analyze all training logs centrally on mteverest4"
}
install_system_packages_on_servers() {
    generate_map
    
    echo "--------------------------------------------------------------------------"
    echo "                    Installing System Packages on Servers                    "    
    echo "--------------------------------------------------------------------------"
    
    password="${password:-$DEFAULT_PASSWORD}"
    
    for key in "${!machine_server_map[@]}"
    do
        destination=$key
        
        echo "Installing system packages on server: $destination"
        
        # Update package list and install python3-venv using echo to provide sudo password
        echo "  Updating package list and installing python3-venv..."
        sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@$destination "echo '$password' | sudo -S apt update && echo '$password' | sudo -S apt install -y python3-venv python3-pip"
        
        # Verify installation
        echo "  Verifying python3-venv installation..."
        sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@$destination "python3 -m venv --help"
        
        echo "  Done with $destination"
        echo
    done
    
    echo "System packages installed on all servers successfully!"
}

close_servers() {
    generate_map
    echo
    echo "------------------------------------"
    echo "$(date) Stopping KeyDB SERVERS:"
    echo "------------------------------------"
    echo

    STARTING_SERVERS_START=$(date +%s)

    server_node_id_counter=0
    
    for key in "${!machine_server_map[@]}"
    do  
        echo "Stopping KeyDB server on: $key"
        destination=$key
        
        # Stop KeyDB and Redis processes
    sshpass -p "$password" ssh -o StrictHostKeyChecking=no -n "$username@$destination" "
            echo '  🛑 Stopping KeyDB server processes...'
            pkill redis 2>/dev/null || echo '    No redis processes found'
            pkill -9 keydb 2>/dev/null || echo '    No keydb processes found'
            echo '  ✅ KeyDB server stopped on $destination'
        "
        
        echo "  Done with $destination"
        echo
    done

    echo "KeyDB servers stopped on all machines!"
}   
start_servers() {
    generate_map
    echo
    echo "------------------------------------------------------------------------------------------------------------------------------------"
    echo "$(date) STARTING SERVERS:"
    echo "------------------------------------------------------------------------------------------------------------------------------------"
    echo

    STARTING_SERVERS_START=$(date +%s)

    server_node_id_counter=0


    
    for key in "${!machine_server_map[@]}"
    do  
        echo "$key"
        destination=$key
         # Extract username from the value
         # Command to execute
        replicas=""
        for replica in ${replicas_map[$key]}; do
            replicas+="--replicaof $replica 6379 "
        done
        replicas="${replicas% }"  # Trim the trailing space
        # echo $replicas
        # echo "./fall_2024/KeyDB/src/keydb-server ./fall_2024/KeyDB/keydb.conf --multi-master yes --active-replica yes  $replicas;"
         echo "STARTING SERVERS: in $destination"
        # Run SSH command with a single block of shell commands
        echo "$username@$destination ./KeyDB/src/keydb-server ./KeyDB/keydb.conf --multi-master yes --active-replica yes --logfile ./KeyDB/$key.log $replicas ;"
    sshpass -p "$password" ssh -o StrictHostKeyChecking=no -n "$username@$destination" "
            cd ~/KeyDB
            nohup ./src/keydb-server ./keydb.conf \
                --multi-master yes \
                --active-replica yes \
                --logfile ./$key.log \
                --port 6379 \
                $replicas > ./$key.out 2>&1 &
            echo \$! > ./$key.pid
            echo 'KeyDB server started on $key with PID: '\$(cat ./$key.pid)
        "
        # ssh -n "$username@$destination" " ./KeyDB/src/keydb-server ./KeyDB/keydb.conf ; " 
        
        # # Check if the SSH command was successful and print output
        # if [ $? -eq 0 ]; then
        #     echo "Server $destination responded with: $output"
        # else
        #     echo "Failed to start server on $destination."
        # fi
    done

   
}   

copy_keydb_config() {
    generate_map
    
    for key in "${!machine_server_map[@]}"
    do
        destination=$key
        echo "Copying config to server: $destination"
        
        # Backup old config first
       
        
        # Copy new config
    sshpass -p "$password" scp -o StrictHostKeyChecking=no /home/abhattar/ml_cars/KeyDB $username@$destination:~/KeyDB/
        
        echo "  ✅ Config copied to $destination"
    done
}


delete_database(){
    generate_map
    echo "--------------------------------------------------------------------------"
    echo "                       Deleting RDB Files on Servers                      "    
    echo "--------------------------------------------------------------------------"
    
    for key in "${!machine_server_map[@]}"
    do
        destination=$key
        
        echo "Deleting RDB files on $key"
        
    sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@$destination "
            cd ~/KeyDB
            echo '  🗑️  Deleting RDB files from KeyDB directory...'
            rm -f *.rdb *.aof dump.rdb
            echo '  ✅ RDB files deleted from KeyDB directory'
        "
        
    done
}

initialize_mlp_weights_on_yangra1() {
    echo "--------------------------------------------------------------------------"
    echo "                    Initializing Weights on yangra1                          "    
    echo "--------------------------------------------------------------------------"
    
    echo "Connecting to yangra1 and initializing weights..."
    
    sshpass -p "$password" ssh -o StrictHostKeyChecking=no abhattar@mteverest4.uwyo.edu "bash -c '
        cd ~/ml_cars
        source myenv/bin/activate
        echo \"✅ Environment activated\"
        echo \"🚀 Running weight initializer...\"
        python weight_intializer.py
        echo \"✅ Weight initialization completed\"
    '"
    
    echo "Weight initialization completed on yangra1!"
}


initialize_cnn_weights_on_yangra1() {
    echo "--------------------------------------------------------------------------"
    echo "                    Initializing Weights on yangra1                          "    
    echo "--------------------------------------------------------------------------"
    
    echo "Connecting to yangra1 and initializing weights..."
    
    sshpass -p "$password" ssh -o StrictHostKeyChecking=no abhattar@mteverest1.uwyo.edu "bash -c '
        cd ~/ml_cars
        source myenv/bin/activate
        echo \"✅ Environment activated\"
        echo \"🚀 Running weight initializer...\"
        python cnn_weights_initializer.py
        echo \"✅ Weight initialization completed\"
    '"
    
    echo "Weight initialization completed on yangra1!"
}
# Initialize CIC-IDS2017 MLP weights on one server
intitialize_cic_weights_on_servers() {
    echo "--------------------------------------------------------------------------"
    echo "                    Initializing CIC Weights on server                        "    
    echo "--------------------------------------------------------------------------"
    
    # Choose a server to run the initializer (any one is fine since KeyDB is distributed)
    target_server="mteverest1"
    echo "Connecting to $target_server and initializing CIC weights..."
    
    sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@$target_server.uwyo.edu "bash -c '
        cd ~/ml_cars
        source myenv/bin/activate
        echo "✅ Environment activated"
        echo "🚀 Running CIC weight initializer..."
        python cic_weight_initializer.py
        echo "✅ CIC weight initialization completed"
    '"
    
    echo "CIC weight initialization completed on $target_server!"
}

# Initialize CIC-IDS2017 CNN weights on one server
initialize_cic_cnn_weights_on_servers() {
    echo "--------------------------------------------------------------------------"
    echo "                    Initializing CIC CNN Weights on server                   "
    echo "--------------------------------------------------------------------------"

    target_server="mteverest1"
    echo "Connecting to $target_server and initializing CIC CNN weights..."

    sshpass -p "$password" ssh -o StrictHostKeyChecking=no $username@$target_server.uwyo.edu "bash -c '
        cd ~/ml_cars
        source myenv/bin/activate
        echo "✅ Environment activated"
        echo "🚀 Running CIC CNN weight initializer..."
        python cic_cnn_weights_initializer.py
        echo "✅ CIC CNN weight initialization completed"
    '"

    echo "CIC CNN weight initialization completed on $target_server!"
}
# Add after the initialize_weights_on_yangra1 function

start_distributed_training() {
    generate_map
    
    echo "--------------------------------------------------------------------------"
    echo "                    Starting Distributed Training on All Servers            "    
    echo "--------------------------------------------------------------------------"
    
    for key in "${!machine_server_map[@]}"
    do
        destination=$key
        echo "Starting distributed training on server: $destination"
        
        # Start training in background on each server
    sshpass -p "$password" ssh -o StrictHostKeyChecking=no -n "$username@$destination" "bash -c '
            cd ~/ml_cars
            mkdir -p logs
            source myenv/bin/activate
            echo \"✅ Environment activated on $destination\"
            echo \"🚀 Starting distributed training...\"
            nohup python distributed_mlp_trainer.py > training_$destination.log 2>&1 &
            echo \$! > training_$destination.pid
            echo \"Training started on $destination with PID: \"\$(cat training_$destination.pid)
        '"
        
        if [ $? -eq 0 ]; then
            echo "  ✅ Training started successfully on $destination"
        else
            echo "  ❌ Failed to start training on $destination"
        fi
        
        echo
    done
    
    echo "Distributed training started on all servers!"
    echo "Check training status with: python training_monitor.py"
}

start_distributed_cic_training() {
    generate_map
    
    echo "--------------------------------------------------------------------------"
    echo "                    Starting CIC Distributed Training on All Servers        "    
    echo "--------------------------------------------------------------------------"
    
    for key in "${!machine_server_map[@]}"
    do
        destination=$key
        echo "Starting CIC distributed training on server: $destination"
        
        # Start training in background on each server; shards placed at ~/cic_shard by copy_cic_dataset_shards_to_servers
    sshpass -p "$password" ssh -o StrictHostKeyChecking=no -n "$username@$destination" "bash -c '
            cd ~/ml_cars
            source myenv/bin/activate
            echo \"✅ Environment activated on $destination\"
            echo \"📦 Installing scikit-learn for metrics calculation...\"
            
            echo \"🚀 Starting CIC distributed training...\"
            mkdir -p logs
            export CIC_TRAIN_FILE=../cic_shard/cic_train.parquet
            export CIC_TEST_FILE=../cic_shard/cic_test.parquet
            nohup python distributed_cic_mlp_trainer.py > training_cic_$destination.log 2>&1 &
            echo \$! > training_cic_$destination.pid
            echo \"CIC training started on $destination with PID: \"\$(cat training_cic_$destination.pid)
        '"
        
        if [ $? -eq 0 ]; then
            echo "  ✅ CIC training started successfully on $destination"
        else
            echo "  ❌ Failed to start CIC training on $destination"
        fi
        
        echo
    done
    
    echo "CIC distributed training started on all servers!"
}


start_seq_distributed_cic_training() {
    generate_map
    
    echo "--------------------------------------------------------------------------"
    echo "                    Starting CIC Sequential Distributed Training on All Servers        "    
    echo "--------------------------------------------------------------------------"
    
    for key in "${!machine_server_map[@]}"
    do
        destination=$key
        echo "Starting CIC sequential distributed training on server: $destination"
        
        # Start training in background on each server; shards placed at ~/cic_shard by copy_cic_dataset_shards_to_servers
    sshpass -p "$password" ssh -o StrictHostKeyChecking=no -n "$username@$destination" "bash -c '
            cd ~/ml_cars
            source myenv/bin/activate
            echo \"✅ Environment activated on $destination\"
            
            echo \"🚀 Starting CIC sequential distributed training...\"
            mkdir -p logs
            export CIC_TRAIN_FILE=../cic_shard/cic_train.parquet
            export CIC_TEST_FILE=../cic_shard/cic_test.parquet
            nohup python seq_distributed_cic_mlp_trainer.py > training_cic_seq_$destination.log 2>&1 &
            echo \$! > training_cic_seq_$destination.pid
            echo \"CIC sequential training started on $destination with PID: \"\$(cat training_cic_seq_$destination.pid)
        '"
        
        if [ $? -eq 0 ]; then
            echo "  ✅ CIC sequential training started successfully on $destination"
        else
            echo "  ❌ Failed to start CIC sequential training on $destination"
        fi
        
        echo
    done
    
    echo "CIC sequential distributed training started on all servers!"
}


start_distributed_cnn_training() {
    generate_map
    
    echo "--------------------------------------------------------------------------"
    echo "                    Starting Distributed Training on All Servers            "    
    echo "--------------------------------------------------------------------------"
    
    for key in "${!machine_server_map[@]}"
    do
        destination=$key
        echo "Starting distributed training on server: $destination"
        
    # Start training in background on each server
    sshpass -p "$password" ssh -o StrictHostKeyChecking=no -n "$username@$destination" "bash -c '
            cd ~/ml_cars
            source myenv/bin/activate
            echo \"✅ Environment activated on $destination\"
            echo \"📦 Installing scikit-learn for metrics calculation...\"
            pip install scikit-learn
            echo \"🚀 Starting distributed training...\"
            nohup python distributed_cnn_trainer.py > training_$destination.log 2>&1 &
            echo \$! > training_$destination.pid
            echo \"Training started on $destination with PID: \"\$(cat training_$destination.pid)
        '"
        
        if [ $? -eq 0 ]; then
            echo "  ✅ Training started successfully on $destination"
        else
            echo "  ❌ Failed to start training on $destination"
        fi
        
        echo
    done
    
    echo "Distributed training started on all servers!"
    echo "Check training status with: python training_monitor.py"
}

start_distributed_cic_cnn_training() {
    generate_map
    
    echo "--------------------------------------------------------------------------"
    echo "                    Starting CIC CNN Distributed Training on All Servers     "    
    echo "--------------------------------------------------------------------------"
    
    for key in "${!machine_server_map[@]}"
    do
        destination=$key
        echo "Starting CIC CNN distributed training on server: $destination"
        
    sshpass -p "$password" ssh -o StrictHostKeyChecking=no -n "$username@$destination" "bash -c '
            cd ~/ml_cars
            source myenv/bin/activate
            echo "✅ Environment activated on $destination"
            echo "🚀 Starting CIC CNN distributed training..."
            mkdir -p logs
            export CIC_TRAIN_FILE=../cic_shard/cic_train.parquet
            export CIC_TEST_FILE=../cic_shard/cic_test.parquet
            nohup python distributed_cic_cnn_trainer.py > training_cic_cnn_$destination.log 2>&1 &
            echo $! > training_cic_cnn_$destination.pid
            echo "CIC CNN training started on $destination with PID: "$(cat training_cic_cnn_$destination.pid)
        '"
        
        if [ $? -eq 0 ]; then
            echo "  ✅ CIC CNN training started successfully on $destination"
        else
            echo "  ❌ Failed to start CIC CNN training on $destination"
        fi
        
        echo
    done
    
    echo "CIC CNN distributed training started on all servers!"
}

start_seq_distributed_cic_cnn_training() {
    generate_map
    
    echo "--------------------------------------------------------------------------"
    echo "               Starting CIC CNN Sequential Distributed Training             "    
    echo "--------------------------------------------------------------------------"
    
    for key in "${!machine_server_map[@]}"
    do
        destination=$key
        echo "Starting CIC CNN sequential training on server: $destination"
        
    sshpass -p "$password" ssh -o StrictHostKeyChecking=no -n "$username@$destination" "bash -c '
            cd ~/ml_cars
            source myenv/bin/activate
            echo "✅ Environment activated on $destination"
            echo "🚀 Starting CIC CNN sequential training..."
            mkdir -p logs
            export CIC_TRAIN_FILE=../cic_shard/cic_train.parquet
            export CIC_TEST_FILE=../cic_shard/cic_test.parquet
            nohup python seq_distributed_cic_cnn_trainer.py > training_cic_cnn_seq_$destination.log 2>&1 &
            echo $! > training_cic_cnn_seq_$destination.pid
            echo "CIC CNN sequential training started on $destination with PID: "$(cat training_cic_cnn_seq_$destination.pid)
        '"
        
        if [ $? -eq 0 ]; then
            echo "  ✅ CIC CNN sequential training started successfully on $destination"
        else
            echo "  ❌ Failed to start CIC CNN sequential training on $destination"
        fi
        
        echo
    done
    
    echo "CIC CNN sequential distributed training started on all servers!"
}
start_seq_distributed_cnn_training() {
    generate_map
    
    echo "--------------------------------------------------------------------------"
    echo "                    Starting Distributed Training on All Servers            "    
    echo "--------------------------------------------------------------------------"
    
    for key in "${!machine_server_map[@]}"
    do
        destination=$key
        echo "Starting distributed training on server: $destination"
        
        # Start training in background on each server
    sshpass -p "$password" ssh -o StrictHostKeyChecking=no -n "$username@$destination" "bash -c '
            cd ~/ml_cars
            source myenv/bin/activate
            echo \"✅ Environment activated on $destination\"
            echo \"📦 Installing scikit-learn for metrics calculation...\"
            pip install scikit-learn
            echo \"🚀 Starting distributed training...\"
            mkdir -p logs
            nohup python seq_distributed_cnn_trainer.py > training_$destination.log 2>&1 &
            echo \$! > training_$destination.pid
            echo \"Training started on $destination with PID: \"\$(cat training_$destination.pid)
        '"
        
        if [ $? -eq 0 ]; then
            echo "  ✅ Training started successfully on $destination"
        else
            echo "  ❌ Failed to start training on $destination"
        fi
        
        echo
    done
    
    echo "Distributed training started on all servers!"
    echo "Check training status with: python training_monitor.py"
}

start_seq_distributed_training() {
    generate_map
    
    echo "--------------------------------------------------------------------------"
    echo "                    Starting Distributed Training on All Servers            "    
    echo "--------------------------------------------------------------------------"
    
    for key in "${!machine_server_map[@]}"
    do
        destination=$key
        echo "Starting distributed training on server: $destination"
        
        # Start training in background on each server
    sshpass -p "$password" ssh -o StrictHostKeyChecking=no -n "$username@$destination" "bash -c '
            cd ~/ml_cars
            source myenv/bin/activate
            echo \"✅ Environment activated on $destination\"
            echo \"📦 Installing scikit-learn for metrics calculation...\"
           
            echo \"🚀 Starting distributed training...\"
            nohup python seq_distributed_mlp_trainer.py > training_$destination.log 2>&1 &
            echo \$! > training_$destination.pid
            echo \"Training started on $destination with PID: \"\$(cat training_$destination.pid)
        '"
        
        if [ $? -eq 0 ]; then
            echo "  ✅ Training started successfully on $destination"
        else
            echo "  ❌ Failed to start training on $destination"
        fi
        
        echo
    done
    
    echo "Distributed training started on all servers!"
    echo "Check training status with: python training_monitor.py"
}

start_training_monitor() {
    echo "--------------------------------------------------------------------------"
    echo "                    Starting Training Monitor                               "    
    echo "--------------------------------------------------------------------------"
    
    echo "Starting training monitor on local machine..."
    
    # Check if training_monitor.py exists
    if [ ! -f "training_monitor.py" ]; then
        echo "❌ training_monitor.py not found! Please create it first."
        return 1
    fi
    
    # Start monitor in foreground (no background, no nohup)
    echo "🚀 Starting training monitor..."
    echo "Monitor will run in foreground. Press Ctrl+C to stop."
    echo ""
    
    # Run monitor directly (no &, no nohup)
    python training_monitor.py --wait 1
}

check_training_status() {
    echo "--------------------------------------------------------------------------"
    echo "                    Checking Training Status                               "    
    echo "--------------------------------------------------------------------------"
    
    echo "Checking training status of all servers..."
    
    # Check if training_monitor.py exists
    if [ ! -f "training_monitor.py" ]; then
        echo "❌ training_monitor.py not found!"
        return 1
    fi
    
    # Run status check
    python training_monitor.py --detailed
}

stop_distributed_training() {
    generate_map
    
    echo "--------------------------------------------------------------------------"
    echo "                    Stopping Distributed Training                          "    
    echo "--------------------------------------------------------------------------"
    
    for key in "${!machine_server_map[@]}"
    do
        destination=$key
        echo "Stopping training on server: $destination"
        
    sshpass -p "$password" ssh -o StrictHostKeyChecking=no "$username@$destination" "
            if [ -f ~/ml_cars/training_$destination.pid ]; then
                pid=\$(cat ~/ml_cars/training_$destination.pid)
                if ps -p \$pid > /dev/null 2>&1; then
                    kill \$pid
                    echo 'Training stopped on $destination'
                else
                    echo 'Training process not running on $destination'
                fi
                rm -f ~/ml_cars/training_$destination.pid
            else
                echo 'No training PID file found on $destination'
            fi
        "
        
        echo "  Done with $destination"
    done
    
    echo "Distributed training stopped on all servers!"
}


# Stop all distributed ML-related processes (training and weight initialization) on all servers
stop_all_distributed_ml_processes() {
    generate_map
    
    echo "--------------------------------------------------------------------------"
    echo "                    Stopping ALL Distributed ML Processes                  "    
    echo "--------------------------------------------------------------------------"
    
    # List of trainer and initializer scripts to terminate
    local patterns=(
        "distributed_mlp_trainer.py"
        "seq_distributed_mlp_trainer.py"
        "distributed_cnn_trainer.py"
        "seq_distributed_cnn_trainer.py"
        "distributed_cic_mlp_trainer.py"
        "seq_distributed_cic_mlp_trainer.py"
        "distributed_cic_cnn_trainer.py"
        "seq_distributed_cic_cnn_trainer.py"
        "weight_intializer.py"
        "cnn_weights_initializer.py"
        "lstm_weights_initializer.py"
        "unsw_weight_initializer.py"
        "cic_weight_initializer.py"
        "cic_cnn_weights_initializer.py"
    )

    for key in "${!machine_server_map[@]}"
    do
        destination=$key
        echo "Stopping ML processes on server: $destination"
        
    sshpass -p "$password" ssh -o StrictHostKeyChecking=no "$username@$destination" "bash -s" <<'REMOTE_EOF'
set -e

echo "  🔍 Finding and stopping trainer processes (by PID files where available)"
for f in ~/ml_cars/training_*.pid ~/ml_cars/training_cic_*.pid; do
  [ -f "$f" ] || continue
  pid=$(cat "$f" 2>/dev/null || true)
  if [ -n "$pid" ] && ps -p "$pid" > /dev/null 2>&1; then
    kill "$pid" 2>/dev/null || true
  fi
  rm -f "$f"
done

echo "  🛑 Killing trainer/initializer python scripts by name"
for p in \
  distributed_mlp_trainer.py \
  seq_distributed_mlp_trainer.py \
  distributed_cnn_trainer.py \
  seq_distributed_cnn_trainer.py \
  distributed_cic_mlp_trainer.py \
  seq_distributed_cic_mlp_trainer.py \
  distributed_cic_cnn_trainer.py \
  seq_distributed_cic_cnn_trainer.py \
  weight_intializer.py \
  cnn_weights_initializer.py \
  lstm_weights_initializer.py \
  unsw_weight_initializer.py \
  cic_weight_initializer.py \
  cic_cnn_weights_initializer.py; do
  pkill -f "$p" 2>/dev/null || true
done

echo "  🧹 Cleanup leftover .pid files (if any)"
rm -f ~/ml_cars/training_*.pid ~/ml_cars/training_cic_*.pid 2>/dev/null || true

echo "  ✅ Done on $(hostname)"
REMOTE_EOF

        echo "  Done with $destination"
        echo
    done
    
    echo "All distributed ML processes have been stopped on all servers!"
}

copy_training_logs_from_servers() {
    generate_map
    
    echo "--------------------------------------------------------------------------"
    echo "                    Copying Training Logs from Servers                      "    
    echo "--------------------------------------------------------------------------"
    
    # Create local logs directory
    local_logs_dir="training_logs_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$local_logs_dir"
    
    password="${password:-$DEFAULT_PASSWORD}"
    
    for key in "${!machine_server_map[@]}"
    do
        destination=$key
        echo "Copying logs from server: $destination"
        
        # Create server-specific subdirectory
        server_logs_dir="$local_logs_dir/$destination"
        mkdir -p "$server_logs_dir"
        
        # Copy training log files
        echo "  Copying training logs..."
        sshpass -p "$password" scp -o StrictHostKeyChecking=no "$username@$destination:~/ml_cars/logs/*.csv" "$server_logs_dir/" 2>/dev/null
        
        # Copy training output logs (generic and CIC-specific)
        echo "  Copying training output logs..."
        sshpass -p "$password" scp -o StrictHostKeyChecking=no "$username@$destination:~/ml_cars/training_*.log" "$server_logs_dir/" 2>/dev/null
        sshpass -p "$password" scp -o StrictHostKeyChecking=no "$username@$destination:~/ml_cars/training_cic_*.log" "$server_logs_dir/" 2>/dev/null
        
        # Copy training PID files (for status checking) (generic and CIC-specific)
        echo "  Copying PID files..."
        sshpass -p "$password" scp -o StrictHostKeyChecking=no "$username@$destination:~/ml_cars/training_*.pid" "$server_logs_dir/" 2>/dev/null
        sshpass -p "$password" scp -o StrictHostKeyChecking=no "$username@$destination:~/ml_cars/training_cic_*.pid" "$server_logs_dir/" 2>/dev/null
        
        # List copied files
        echo "  Files copied to $server_logs_dir:"
        ls -la "$server_logs_dir/" 2>/dev/null || echo "    No files found"
        
        echo "  Done with $destination"
        echo
    done
    
    echo "✅ Training logs copied to: $local_logs_dir"
    echo "📁 Each server has its own subdirectory with logs"
    echo "📊 You can now analyze training progress locally"
}


# The given function below can be used for Distributed ML training in sequential or eventual mode 

# You 
train_1()
{
    close_servers
    delete_database
    delete_keydb_logs

    copy_distributed_trainer_to_servers
    stop_all_distributed_ml_processes
    start_servers

    sleep 10 
    

    initialize_cnn_weights_on_yangra1
    sleep 10



   
    start_seq_distributed_cnn_training  #  sequential distributed CNN training
    # start_distributed_cnn_training # Uncomment to run eventual consistency training 
    sleep 150m

    close_servers
    copy_keydb_logs
    copy_logs
}




train_2(){
    close_servers
    delete_database
    delete_keydb_logs

    copy_distributed_trainer_to_servers
    stop_all_distributed_ml_processes
    start_servers

    sleep 10 
    # initialize_mlp_weights_on_yangra1

    initialize_cnn_weights_on_yangra1
    sleep 10



    # start_distributed_cic_training
    # start_distributed_cic_cnn_training
    # close_servers
    start_distributed_cnn_training
    sleep 1600

    close_servers
    copy_keydb_logs
    copy_logs
}






train_2





# copy_keydb_logs