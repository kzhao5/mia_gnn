#!/bin/bash

py_path=$(which python)

run_threshold_mia() {
    number=$1
    for i in $(seq 1 "$number"); do
        echo "Running Threshold-based MIA attack, iteration $i of $number"
        $py_path /home/kzhao/MIA-GNN/threshold_based_attack.py
    done
}

if [ "$1" == "--number" ]; then
    if [ -z "$2" ] || ! [[ "$2" =~ ^[0-9]+$ ]]; then
        echo "Error: Please provide a valid number of runs."
        echo "Usage: $0 --number <number_of_runs>"
        exit 1
    fi
    run_threshold_mia "$2"
else
    echo "Usage: $0 --number <number_of_runs>"
    exit 1
fi