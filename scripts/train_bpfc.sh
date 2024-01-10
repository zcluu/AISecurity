#!/bin/bash

clear

ip_port=$(od -An -N2 -i /dev/urandom | awk '{print ($1 % 5001) + 15000}')
echo "port: $ip_port"
current_time=$(date "+%Y-%m-%d_%H-%M-%S")
save_dir="results/bpfc/$current_time"
mkdir -p "$save_dir"

python_command="python tools/train_defense.py --save_dir $save_dir --port $ip_port"

echo "command: $python_command";

nohup $python_command > "$save_dir/run.log" 2>&1 &

watch -n 0 tail -n 30 "$save_dir/run.log"