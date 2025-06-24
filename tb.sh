#!/bin/bash

base1="/home/astwea/MyDogTask/Mydog/logs/rl_games/diff_drive_direct/"
base2="/home/astwea/MyDogTask/Mydog/runs/logs/"

latest1=$(find "$base1" -mindepth 2 -type d -name "summaries" | sort | tail -n 1)
latest2=$(find "$base2" -mindepth 2 -type d -name "summary" | sort | tail -n 1)

port1=6006
port2=6007

# 构建参数（有多少tab就写多少）
args=()
if [ -n "$latest1" ]; then
    args+=(--title="TB1" -e "bash -c 'echo TensorBoard $latest1 on $port1; tensorboard --logdir=\"$latest1\" --port=$port1; exec bash'")
fi
if [ -n "$latest2" ]; then
    args+=(--tab --title="TB2" -e "bash -c 'echo TensorBoard $latest2 on $port2; tensorboard --logdir=\"$latest2\" --port=$port2; exec bash'")
fi

if [ "${#args[@]}" -gt 0 ]; then
    gnome-terminal --window "${args[@]}"
fi
