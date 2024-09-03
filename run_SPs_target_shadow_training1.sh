# #!/bin/bash

# # Default values
# NUMBER=1
# START_EPOCH=100
# DATASET="MNIST"

# # Parse command line arguments
# while [[ $# -gt 0 ]]
# do
# key="$1"

# case $key in
#     --number)
#     NUMBER="$2"
#     shift # past argument
#     shift # past value
#     ;;
#     --start_epoch)
#     START_EPOCH="$2"
#     shift # past argument
#     shift # past value
#     ;;
#     --dataset)
#     DATASET="$2"
#     shift # past argument
#     shift # past value
#     ;;
#     *)    # unknown option
#     echo "Unknown option: $1"
#     exit 1
#     ;;
# esac
# done

# # Print the parameters
# echo "Number: $NUMBER"
# echo "Start Epoch: $START_EPOCH"
# echo "Dataset: $DATASET"

# # Run the Python script with the provided arguments
# python code/main_SPs_graph_classification1.py \
#     --config '/home/kzhao/mia_gnn/configs/SPS/superpixels_graph_classification_GCN_CIFAR10_100k.json' \
#     --dataset "$DATASET" \
#     --gpu_id 0 \
#     --model GatedGCN \
#     --out_dir 'out/SPs_graph_classification/' \
#     --epochs $START_EPOCH \
#     --batch_size 32 \
#     --init_lr 0.001 \
#     # --num_workers 8

#!/bin/bash

# Default values
NUMBER=1
START_EPOCH=100
DATASET="MNIST"

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
python code/main_SPs_graph_classification2.py \
    --config '/home/kzhao/mia_gnn/configs/SPS/superpixels_graph_classification_GCN_CIFAR10_100k.json' \
    --dataset "$DATASET" \
    --gpu_id 0 \
    --model GatedGCN \
    --out_dir 'out/SPs_graph_classification/' \
    --epochs $START_EPOCH \
    --batch_size 32 \
    --init_lr 0.001 \
    # --num_workers 8