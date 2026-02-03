#!/bin/bash

# File containing the list of remote nodes and their passwords
NODES_FILE="nodes.txt"

# Path to the SSH key (default location)
SSH_KEY="$HOME/.ssh/id_rsaz"  # Remove the 'z' at the end

# Ensure SSH key exists
if [ ! -f "$SSH_KEY" ]; then
    echo "SSH key not found at $SSH_KEY. Generating a new one..."
    ssh-keygen -t rsa -b 4096 -N "" -f "$SSH_KEY"
fi

# Check if nodes file exists
if [ ! -f "$NODES_FILE" ]; then
    echo "Error: $NODES_FILE not found!"
    exit 1
fi

# Ensure sshpass is installed
if ! command -v sshpass &>/dev/null; then
    echo "sshpass is not installed. Installing it is required for this script."
    exit 1
fi

# Loop through each node and password in the file
while read -r NODE PASSWORD; do
    # Skip empty lines and comments
    if [[ -z "$NODE" || "$NODE" == \#* ]]; then
        continue
    fi

    echo "Setting up password-less SSH for $NODE..."

    # Use sshpass to copy the SSH key to the remote node
    sshpass -p "$PASSWORD" ssh-copy-id -i "$SSH_KEY" "$NODE" 2>/dev/null

    # Check if the copy was successful
    if [ $? -eq 0 ]; then
        echo "Password-less SSH set up for $NODE successfully."
    else
        echo "Failed to set up password-less SSH for $NODE. Check connection or credentials."
    fi
done < "$NODES_FILE"

echo "Password-less SSH setup complete for all nodes listed in $NODES_FILE."
