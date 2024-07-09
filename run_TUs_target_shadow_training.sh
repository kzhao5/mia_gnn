#!/bin/bash

py_path=$(which python)

# 初始化变量
number=1
start_epoch=100
dataset="DD"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --number)
        number="$2"
        shift 2
        ;;
        --start_epoch)
        start_epoch="$2"
        shift 2
        ;;
        --dataset)
        dataset="$2"
        shift 2
        ;;
        *)    # unknown option
        shift
        ;;
    esac
done

# 运行函数
run() {
    for i in $(seq 1 $number); do
        epoch=$((i * start_epoch))
        $py_path code/main_TUs_graph_classification.py --dataset $dataset --config "configs/TUS/TUs_graph_classification_GCN_${dataset}_100k.json" --epochs $epoch
    done
}

# 执行运行函数
run