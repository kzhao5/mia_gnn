# Membership Inference Attacks to GNN for Graph Classification

The source code for ICDM2021 paper: "Adapting Membership Inference Attacks to GNN for Graph Classification: Approaches and Implications".
The full version of the paper can be found in [https://arxiv.org/abs/2110.08760](https://arxiv.org/abs/2110.08760)


# Installation

If you meet the version mismatch error for **Lasagne** library, please use following command to
upgrade **Lasagne** library.
<pre>
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
</pre>

# Usage

## Step 1: Train Target and Shadow Model


To train the victim GCN model, use the following command. 


For common graph dataset (DD, PROTEINS_full, ENZYMES):
<pre>
bash run_TUs_target_shadow_training.sh --number 10 --start_epoch 100 --dataset DD
</pre>
./run_TUs_target_shadow_training.sh --number 10 --start_epoch 100 --dataset DD --pretrain_epochs 50 --use_pretrained

./run_TUs_target_shadow_training.sh --number 1 --start_epoch 100 --dataset DD --pretrain_epochs 100 --use_pretrained

For graph converted via SuperPixel (CIFAR10, MNIST)
<pre>
bash run_SPs_target_shadow_training.sh --number 10 --start_epoch 100 --dataset MNIST
</pre>

bash run_SPs_target_shadow_training.sh --number 1 --start_epoch 100 --dataset CIFAR10

--number this is the number of repeated model training 

--start_epoch this is the minimum number of interactions to train the model 

--dataset this is the dataset name of model training


For detailed code execution, you can refer to 'main_SPs_graph_classification.py' and 'main_TUs_graph_classification.py' in the ./code folder.



## Step 2: Train Attack Model Inferring Graph Sample Membership

<pre>
bash run_transfer_attach.sh --number 15
</pre>

--number this is the number of repeated attack


For detailed code execution, you can refer to 'transfer_based_attack.py'.

<pre>
srun python3 -m venv ~/mia_gnn
source ~/mia_gnn_env/bin/activate

pip install dgl==2.0.0 -f https://data.dgl.ai/wheels/cu118/repo.html
<pre>