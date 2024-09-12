# # shellcheck disable=SC2006
# py_path=`which python`
# start_epoch=$2
# dataset=$3
# run() {
#     number=$1
#     shift
#     for i in $(seq $number); do
#       # shellcheck disable=SC2068
#       $@
#       epoch=`expr $i \* $start_epoch`
#       $py_path code/main_SPs_graph_classification.py --dataset $dataset --config 'configs/SPS/superpixels_graph_classification_GCN_'$dataset'_100k.json' --epochs $epoch
#     done
# }

# # shellcheck disable=SC2046
# # shellcheck disable=SC2006
# #echo $epoch
# run "$1"

#!/bin/bash

# Default values
NUMBER=1
START_EPOCH=100
DATASET="CIFAR10"

# Parse command line arguments
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --number)
    NUMBER="$2"
    shift # past argument
    shift # past value
    ;;
    --start_epoch)
    START_EPOCH="$2"
    shift # past argument
    shift # past value
    ;;
    --dataset)
    DATASET="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    echo "Unknown option: $1"
    exit 1
    ;;
esac
done

# Print the parameters
echo "Number: $NUMBER"
echo "Start Epoch: $START_EPOCH"
echo "Dataset: $DATASET"

# Run the Python script with the provided arguments
python code/main_SPs_graph_classification.py \
    --config '/home/kzhao/mia_gnn/configs/SPS/superpixels_graph_classification_GatedGCN_CIFAR10_100k.json' \
    --dataset "$DATASET" \
    --gpu_id 0 \
    --model GatedGCN \
    --out_dir 'out/SPs_graph_classification/' \
    --epochs $START_EPOCH \
    --batch_size 32 \
    --init_lr 0.001 \
    # --num_workers 8