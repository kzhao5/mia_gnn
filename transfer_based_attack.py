import numpy as np
import torch
import random
import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score, f1_score, roc_auc_score
import warnings
import os
from attack_models import MLP
from utils import load_pickled_data, select_top_k, binary_acc, testData, trainData
warnings.simplefilter("ignore")

# NEW: 导入绘图库
import matplotlib.pyplot as plt
import seaborn as sns

def get_latest_run(base_path):
    runs = glob.glob(os.path.join(base_path, 'GatedGCN_CIFAR10_GPU*'))
    return max(runs, key=os.path.getctime) if runs else None

def load_data(run_path, run_type):
    X_train_in = load_pickled_data(os.path.join(run_path, f'{run_type}_X_train_Label_1.pickle'))
    y_train_in = load_pickled_data(os.path.join(run_path, f'{run_type}_y_train_Label_1.pickle'))
    X_train_out = load_pickled_data(os.path.join(run_path, f'{run_type}_X_train_Label_0.pickle'))
    y_train_out = load_pickled_data(os.path.join(run_path, f'{run_type}_y_train_Label_0.pickle'))
    num_node_1 = load_pickled_data(os.path.join(run_path, f'{run_type}_num_node_1.pickle'))
    num_node_0 = load_pickled_data(os.path.join(run_path, f'{run_type}_num_node_0.pickle'))
    num_edge_1 = load_pickled_data(os.path.join(run_path, f'{run_type}_num_edge_1.pickle'))
    num_edge_0 = load_pickled_data(os.path.join(run_path, f'{run_type}_num_edge_0.pickle'))
    return X_train_in, y_train_in, X_train_out, y_train_out, num_node_1, num_node_0, num_edge_1, num_edge_0

def transfer_based_attack(epochs):
    base_path = '/home/kzhao/mia_gnn/out/SPs_graph_classification/checkpoints'
    latest_run = get_latest_run(base_path)
    if not latest_run:
        raise FileNotFoundError(f"No data found in {base_path}")

    print(f"Using data from: {latest_run}")

    S_X_train_in, S_y_train_in, S_X_train_out, S_y_train_out, S_Label_1_num_nodes, S_Label_0_num_nodes, S_Label_1_num_edges, S_Label_0_num_edges = load_data(os.path.join(latest_run, 'S_RUN_'), 'S')
    T_X_train_in, T_y_train_in, T_X_train_out, T_y_train_out, T_Label_1_num_nodes, T_Label_0_num_nodes, T_Label_1_num_edges, T_Label_0_num_edges = load_data(os.path.join(latest_run, 'T_RUN_'), 'T')

    X_attack = torch.FloatTensor(np.concatenate((S_X_train_in, S_X_train_out), axis=0))
    X_attack_nodes = torch.FloatTensor(np.concatenate((S_Label_1_num_nodes, S_Label_0_num_nodes), axis=0))
    X_attack_edges = torch.FloatTensor(np.concatenate((S_Label_1_num_edges, S_Label_0_num_edges), axis=0))

    y_target = torch.FloatTensor(np.concatenate((T_y_train_in, T_y_train_out), axis=0))
    y_attack = torch.FloatTensor(np.concatenate((S_y_train_in, S_y_train_out), axis=0))
    X_target = torch.FloatTensor(np.concatenate((T_X_train_in, T_X_train_out), axis=0))
    X_target_nodes = torch.FloatTensor(np.concatenate((T_Label_1_num_nodes, T_Label_0_num_nodes), axis=0))
    X_target_edges = torch.FloatTensor(np.concatenate((T_Label_1_num_edges, T_Label_0_num_edges), axis=0))

    feature_nums = min(X_attack.shape[1], X_target.shape[1])
    selected_X_target = select_top_k(X_target, feature_nums)
    selected_X_attack = select_top_k(X_attack, feature_nums)

    n_in = selected_X_attack.shape[1]
    attack_model = MLP(in_size=n_in, out_size=1, hidden_1=64, hidden_2=64)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.0001)
    attack_data = trainData(selected_X_attack, y_attack)
    target_data = testData(selected_X_target)
    train_loader = DataLoader(dataset=attack_data, batch_size=64, shuffle=True)
    target_loader = DataLoader(dataset=target_data, batch_size=1)

    for i in range(epochs):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = attack_model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    # NEW: 初始化用于存储置信度的列表
    y_pred_list = []
    y_pred_prob_list = []
    member_confidences = []
    non_member_confidences = []

    attack_model.eval()
    with torch.no_grad():
        for X_batch, num_node, num_edge, y in zip(target_loader, X_target_nodes, X_target_edges, y_target):
            y_test_pred = attack_model(X_batch)
            y_test_pred_prob = torch.sigmoid(y_test_pred)
            y_pred_prob_list.append(y_test_pred_prob.item())
            y_pred_tag = torch.round(y_test_pred_prob)
            y_pred_list.append(y_pred_tag.cpu().numpy()[0])

            # NEW: 收集成员和非成员的置信度
            if y.item() == 1:
                member_confidences.append(y_test_pred_prob.item())
            else:
                non_member_confidences.append(y_test_pred_prob.item())

    # NEW: 计算评估指标
    accuracy = accuracy_score(y_target, y_pred_list)
    precision, recall, f1, _ = precision_recall_fscore_support(y_target, y_pred_list, average='macro')
    auc = roc_auc_score(y_target, y_pred_prob_list)

    # NEW: 打印结果
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

    # NEW: 绘制置信度分布图
    plt.figure(figsize=(10, 6))
    sns.histplot(member_confidences, kde=True, stat="density", alpha=0.5, label="Member", color="red")
    sns.histplot(non_member_confidences, kde=True, stat="density", alpha=0.5, label="Non-member", color="blue")
    plt.xlabel("Confidence Value")
    plt.ylabel("Density")
    plt.title("MIA Confidence Value Distribution")
    plt.legend()
    plt.savefig("mia_distribution.png")
    plt.close()

    return accuracy, precision, recall, f1, auc

if __name__ == '__main__':
    accuracy, precision, recall, f1, auc = transfer_based_attack(300)
    print(f"Final results: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")