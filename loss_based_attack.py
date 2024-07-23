import os
import pickle
import dgl
import sys
import glob
import torch
import torch.cuda
import types
import torch.nn.functional as F
import numpy as np
from torch import nn
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score
sys.path.append('/home/kzhao/MIA-GNN/code')
from nets.TUs_graph_classification.load_net import gnn_model
from dgl.data import TUDataset
from data.TUs import TUsDataset

def load_pickled_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_data(path, prefix):
    X_train_in = load_pickled_data(os.path.join(path, f'{prefix}_X_train_Label_1.pickle'))
    y_train_in = load_pickled_data(os.path.join(path, f'{prefix}_y_train_Label_1.pickle'))
    X_train_out = load_pickled_data(os.path.join(path, f'{prefix}_X_train_Label_0.pickle'))
    y_train_out = load_pickled_data(os.path.join(path, f'{prefix}_y_train_Label_0.pickle'))
    Label_1_num_nodes = load_pickled_data(os.path.join(path, f'{prefix}_num_node_1.pickle'))
    Label_0_num_nodes = load_pickled_data(os.path.join(path, f'{prefix}_num_node_0.pickle'))
    Label_1_num_edges = load_pickled_data(os.path.join(path, f'{prefix}_num_edge_1.pickle'))
    Label_0_num_edges = load_pickled_data(os.path.join(path, f'{prefix}_num_edge_0.pickle'))
    return X_train_in, y_train_in, X_train_out, y_train_out, Label_1_num_nodes, Label_0_num_nodes, Label_1_num_edges, Label_0_num_edges

# def load_model(model_path, net_params, MODEL_NAME):
#     model = gnn_model(MODEL_NAME, net_params)
#     model.load_state_dict(torch.load(model_path))
#     return model

def load_model(model_path, net_params, MODEL_NAME, device):
    model = gnn_model(MODEL_NAME, net_params)
    state_dict = torch.load(model_path, map_location=device)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k in model.state_dict():
            if v.shape == model.state_dict()[k].shape:
                new_state_dict[k] = v.to(device)
            else:
                print(f"Shape mismatch for {k}: saved {v.shape}, current {model.state_dict()[k].shape}")
                if k == 'embedding_h.weight' or k == 'embedding_h.bias':
                    new_state_dict[k] = model.state_dict()[k].to(device)
                elif k == 'MLP_layer.FC_layers.2.weight' or k == 'MLP_layer.FC_layers.2.bias':
                    new_state_dict[k] = model.state_dict()[k].to(device)
    
    model.load_state_dict(new_state_dict, strict=False)
    return model.to(device)

def compute_loss(model, g, labels, idx, device):
    model.eval()
    with torch.no_grad():
        try:
            sub_g = g.subgraph([idx])
            sub_feat = sub_g.ndata['feat']
            # print(f"Computing loss for idx: {idx}")
            # print(f"sub_g: {sub_g}")
            # print(f"sub_feat shape: {sub_feat.shape}")
            logits = model(sub_g, sub_feat, None)
            # print(f"Logits shape: {logits.shape}")
            # print(f"Logits: {logits}")
            # print(f"Labels shape: {labels[idx:idx+1].shape}")
            # print(f"Label: {labels[idx:idx+1]}")
            
            # 检查标签是否在有效范围内
            if labels[idx] >= logits.shape[1]:
                print(f"Warning: Label {labels[idx]} is out of range for logits with {logits.shape[1]} classes")
                return float('inf')  # 返回一个大的损失值
            
            loss = F.cross_entropy(logits, labels[idx:idx+1])
            print(f"Computed loss: {loss.item()}")
            return loss.item()
        except Exception as e:
            print(f"Error in compute_loss for idx {idx}: {e}")
            return float('inf')  # 返回一个大的损失值
# def compute_loss(model, g, labels, idx, device):
#     model.eval()
#     with torch.no_grad():
#         try:
#             sub_g = g.subgraph([idx])
#             sub_feat = sub_g.ndata['feat']
#             logits = model(sub_g, sub_feat, None)
            
#             # 确保标签是二进制的
#             binary_label = torch.tensor([1 if labels[idx] > 0 else 0], device=device)
            
#             # 如果模型输出是二维的，取第一维
#             if logits.dim() > 1:
#                 logits = logits.squeeze(0)
            
#             loss = F.binary_cross_entropy_with_logits(logits, binary_label.float())
#             return loss.item()
#         except Exception as e:
#             print(f"Error in compute_loss for idx {idx}: {e}")
#             return float('inf')


# def loss_based_mia(model, g, labels, all_nodes, threshold, device):
#     predictions = []
#     losses = []
#     for idx in all_nodes:
#         loss = compute_loss(model, g, labels, idx, device)
#         if loss == float('inf'):
#             predictions.append(0)  # 假设无法计算损失的节点不在训练集中
#         else:
#             predictions.append(1 if loss < threshold else 0)
#         losses.append(loss)
#     return predictions, losses

def loss_based_mia(model, g, labels, all_nodes, threshold, device):
    predictions = []
    losses = []
    for idx in all_nodes:
        loss = compute_loss(model, g, labels, idx, device)
        if loss == float('inf'):
            predictions.append(0)  # 假设无法计算损失的节点不在训练集中
        else:
            predictions.append(1 if loss < threshold else 0)
        losses.append(loss)
    return predictions, losses
# def determine_threshold(model, g, labels, known_train_nodes, known_test_nodes, device):
#     train_losses = []
#     test_losses = []
#     for idx in known_train_nodes:
#         loss = compute_loss(model, g, labels, idx, device)
#         if loss != float('inf'):
#             train_losses.append(loss)
#     for idx in known_test_nodes:
#         loss = compute_loss(model, g, labels, idx, device)
#         if loss != float('inf'):
#             test_losses.append(loss)
    
#     if not train_losses or not test_losses:
#         print("Warning: No valid losses computed for train or test set")
#         return 0.5  # 返回一个默认阈值
    
#     threshold = (np.mean(train_losses) + np.mean(test_losses)) / 2
#     return threshold
def determine_threshold(model, g, labels, known_train_nodes, known_test_nodes, device):
    train_losses = []
    test_losses = []
    print(f"Number of known train nodes: {len(known_train_nodes)}")
    print(f"Number of known test nodes: {len(known_test_nodes)}")
    
    for idx in known_train_nodes:
        loss = compute_loss(model, g, labels, idx, device)
        if loss != float('inf'):
            train_losses.append((loss, 1))  # 1 表示在训练集中
    
    for idx in known_test_nodes:
        loss = compute_loss(model, g, labels, idx, device)
        if loss != float('inf'):
            test_losses.append((loss, 0))  # 0 表示不在训练集中
    
    all_losses = train_losses + test_losses
    print(f"Number of valid losses: {len(all_losses)}")
    
    if not all_losses:
        print("Warning: No valid losses found. Using default threshold.")
        return 0.5  # 使用默认阈值
    
    losses, true_labels = zip(*all_losses)
    
    fpr, tpr, thresholds = roc_curve(true_labels, losses)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold

def get_latest_run(base_path):
    runs = glob.glob(os.path.join(base_path, 'GCN_DD_GPU*'))
    return max(runs, key=os.path.getctime) if runs else None

def print_model_structure(model_path):
    state_dict = torch.load(model_path)
    print("Model structure:")
    for key, value in state_dict.items():
        print(f"{key}: {value.shape}")

# 修改模型的 forward 方法
# def new_forward(self, h):
#     h = self.embedding_h(h)
#     for conv in self.layers:
#         h = conv(h)
#     h = self.MLP_layer(h)
#     return h

def main():
    base_path = '/home/kzhao/MIA-GNN/results/TUs_graph_classification/checkpoints'
    MODEL_NAME = 'GCN'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 清理 CUDA 缓存并设置内存分配器
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.8)  # 使用 80% 的可用 GPU 内存

    # 获取最新的运行结果
    latest_run = get_latest_run(base_path)
    if not latest_run:
        raise FileNotFoundError(f"No data found in {base_path}")
    
    exp_name = os.path.basename(latest_run)
    print(f"Processing latest experiment: {exp_name}")
    
    # 加载DD数据集（下游任务）
    dataset = TUDataset(name='DD')
    graph, _ = dataset[0]
    
    # 加载target model数据
    
    T_X_train_in, T_y_train_in, T_X_train_out, T_y_train_out, T_Label_1_num_nodes, T_Label_0_num_nodes, T_Label_1_num_edges, T_Label_0_num_edges = load_data(os.path.join(latest_run, 'T_RUN_'), 'T')
    
    # 准备数据集
    features = torch.FloatTensor(np.concatenate((T_X_train_in, T_X_train_out), axis=0)).to(device)
    labels = torch.LongTensor(np.concatenate((T_y_train_in, T_y_train_out), axis=0)).to(device)
    num_nodes = torch.FloatTensor(np.concatenate((T_Label_1_num_nodes, T_Label_0_num_nodes), axis=0)).to(device)
    num_edges = torch.FloatTensor(np.concatenate((T_Label_1_num_edges, T_Label_0_num_edges), axis=0)).to(device)
    

    
    # 设置网络参数
    net_params = {
        # 'in_dim': graph.ndata['node_labels'].shape[1],
        # 'hidden_dim': 64,
        # 'out_dim': dataset.num_classes,
        # 'n_classes': dataset.num_classes,
        # 'in_feat_dropout': 0.0,
        # 'dropout': 0.0,
        # 'L': 4,
        # 'readout': 'mean',
        # 'graph_norm': True,
        # 'batch_norm': True,
        # 'residual': True,
        # 'device': device
        'in_dim': features.shape[1],
        'hidden_dim': 138,
        'out_dim': 138,
        'n_classes': dataset.num_classes,  
        'in_feat_dropout': 0.0,
        'dropout': 0.0,
        'L': 4,
        'readout': 'mean',
        'graph_norm': True,
        'batch_norm': True,
        'residual': True,
        'device': device
    }
    
    # 加载微调后的target model
    t_run_path = os.path.join(latest_run, 'T_RUN_')
    model_path = os.path.join(t_run_path, 'target_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = load_model(model_path, net_params, MODEL_NAME, device)
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False  # 禁用梯度计算以节省内存
    # model.forward = types.MethodType(new_forward, model)
    # 创建一个包含所有节点的图
 
    num_nodes = features.shape[0]
    g = dgl.graph((torch.arange(num_nodes), torch.arange(num_nodes))).to(device)
    g.ndata['feat'] = features
    g.ndata['id'] = torch.arange(num_nodes).to(device)

    state_dict = torch.load(model_path, map_location=device)
    
    # 处理形状不匹配的问题
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)

    # 修改模型的嵌入层和MLP层以匹配当前任务
    model.embedding_h = nn.Linear(features.shape[1], net_params['hidden_dim']).to(device)
    model.MLP_layer.FC_layers[-1] = nn.Linear(model.MLP_layer.FC_layers[-1].in_features, 2).to(device)

    print(f"Features device: {features.device}")
    print(f"Labels device: {labels.device}")
    print(f"Model device: {next(model.parameters()).device}")

    print("Testing model...")
    test_loss = compute_loss(model, g, labels, 0, device)  # 直接传递整数 0
    print(f"Test loss: {test_loss}")
    # 打印模型结构，确认所有参数都在正确的设备上
    print("Model structure:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}, device: {param.device}")

    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Number of nodes in graph: {g.number_of_nodes()}")
    print(f"Number of edges in graph: {g.number_of_edges()}")

    # 确保模型的输入维度正确
    if hasattr(model, 'embedding_h'):
        model.embedding_h = nn.Linear(features.shape[1], model.embedding_h.weight.shape[0]).to(device)


     # 如果模型的最后一层不匹配，我们可以手动替换它
    if hasattr(model, 'MLP_layer') and model.MLP_layer.FC_layers[-1].out_features != dataset.num_classes:
        model.MLP_layer.FC_layers[-1] = nn.Linear(model.MLP_layer.FC_layers[-1].in_features, dataset.num_classes).to(device)
    
    # 假设前半部分为已知节点，用于确定阈值
    known_train_nodes = list(range(len(T_X_train_in) // 2))
    known_test_nodes = list(range(len(T_X_train_in), len(T_X_train_in) + len(T_X_train_out) // 2))

    print(f"Number of known train nodes: {len(known_train_nodes)}")
    print(f"Number of known test nodes: {len(known_test_nodes)}")
    
    test_idx = known_train_nodes[0]
    sub_g = g.subgraph([test_idx])
    sub_feat = sub_g.ndata['feat']
    with torch.no_grad():
        test_output = model(sub_g, sub_feat, None)
    print(f"Model output shape: {test_output.shape}")
    print(f"Model output: {test_output}")

    # 确定阈值
    threshold = determine_threshold(model, g, labels, known_train_nodes, known_test_nodes, device)
    print(f"Determined threshold: {threshold}")
   # 检查标签的范围
    print(f"Unique labels: {torch.unique(labels)}")
    print(f"Min label: {labels.min()}, Max label: {labels.max()}")

    # 检查模型的输出维度
    dummy_input = features[:1].to(device)
    dummy_graph = dgl.graph(([0], [0])).to(device)
    dummy_graph.ndata['feat'] = dummy_input
    with torch.no_grad():
        dummy_output = model(dummy_graph, dummy_input, None)
    print(f"Model output dimension: {dummy_output.shape[1]}")

    # 对所有节点执行MIA
 
    all_nodes = list(range(len(features)))
    predictions, losses = loss_based_mia(model, g, labels, all_nodes, threshold, device)
    
    # 评估结果
    true_labels = [1] * len(T_X_train_in) + [0] * len(T_X_train_out)
    accuracy = accuracy_score(true_labels, predictions)
    print(f"MIA Accuracy for {exp_name}: {accuracy}")
    

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    print(f"MIA Results for {exp_name}:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    # # 保存结果
    # results_df = pd.DataFrame({
    #     'node_index': all_nodes,
    #     'true_label': true_labels,
    #     'predicted_label': predictions,
    #     'loss': losses,
    #     'num_nodes': num_nodes.cpu().numpy(),
    #     'num_edges': num_edges.cpu().numpy()
    # })
    # results_df.to_csv(f"mia_results_{exp_name}.csv", index=False)

if __name__ == "__main__":
    main()