# # import numpy as np
# # import torch
# # import random
# # import glob
# # from sklearn.model_selection import train_test_split
# # from torch.utils.data import DataLoader
# # from sklearn.metrics import classification_report, precision_recall_fscore_support
# # import warnings
# # import os
# # from attack_models import MLP
# # from utils import load_pickled_data, select_top_k, binary_acc, testData, trainData
# # warnings.simplefilter("ignore")


# # def transfer_based_attack(epochs):
# #     # GCN_DD_GPU1_12h53m37s_on_Jan_28_2021 0.7900280269	0.6378787879
# #     # GCN_DD_GPU0_19h36m32s_on_Jan_27_2021  0.822117084 0.7315151515

# #     # GCN_PROTEINS_full_GPU0_03h11m51s_on_Jan_28_2021 0.7707677769	0.5766666667
# #     attack_base_path = 'data/statis/GCN/GCN_ENZYMES_GPU0_16h40m29s_on_Jun_08_2021/'
# #     target_base_path = 'data/statis/GCN/GCN_DD_GPU1_16h26m32s_on_Jun_08_2021/'
# #     # GCN_ENZYMES_GPU0_16h40m29s_on_Jun_08_2021 -> GCN_DD_GPU1_16h26m32s_on_Jun_08_2021
# #     # For attack dataset
# #     if os.listdir(attack_base_path).__contains__("S_RUN_"):
# #         S_X_train_in = load_pickled_data(attack_base_path + 'S_RUN_/S_X_train_Label_1.pickle')
# #         S_y_train_in = load_pickled_data(attack_base_path + 'S_RUN_/S_y_train_Label_1.pickle')
# #         S_X_train_out = load_pickled_data(attack_base_path + 'S_RUN_/S_X_train_Label_0.pickle')
# #         S_y_train_out = load_pickled_data(attack_base_path + 'S_RUN_/S_y_train_Label_0.pickle')
# #         S_Label_0_num_nodes = load_pickled_data(attack_base_path + 'S_RUN_/S_num_node_0.pickle')
# #         S_Label_1_num_nodes = load_pickled_data(attack_base_path + 'S_RUN_/S_num_node_1.pickle')
# #         S_Label_0_num_edges = load_pickled_data(attack_base_path + 'S_RUN_/S_num_edge_0.pickle')
# #         S_Label_1_num_edges = load_pickled_data(attack_base_path + 'S_RUN_/S_num_edge_1.pickle')
# #     else:
# #         S_X_train_in = load_pickled_data(attack_base_path + 'X_train_Label_1.pickle')
# #         S_y_train_in = load_pickled_data(attack_base_path + 'y_train_Label_1.pickle')
# #         S_X_train_out = load_pickled_data(attack_base_path + 'X_train_Label_0.pickle')
# #         S_y_train_out = load_pickled_data(attack_base_path + 'y_train_Label_0.pickle')
# #     # For target Dataset
# #     if os.listdir(target_base_path).__contains__("T_RUN_"):
# #         T_X_train_in = load_pickled_data(target_base_path + 'T_RUN_/T_X_train_Label_1.pickle')
# #         T_y_train_in = load_pickled_data(target_base_path + 'T_RUN_/T_y_train_Label_1.pickle')
# #         T_X_train_out = load_pickled_data(target_base_path + 'T_RUN_/T_X_train_Label_0.pickle')
# #         T_y_train_out = load_pickled_data(target_base_path + 'T_RUN_/T_y_train_Label_0.pickle')
# #         T_Label_0_num_nodes = load_pickled_data(target_base_path + 'T_RUN_/T_num_node_0.pickle')
# #         T_Label_1_num_nodes = load_pickled_data(target_base_path + 'T_RUN_/T_num_node_1.pickle')
# #         T_Label_0_num_edges = load_pickled_data(target_base_path + 'T_RUN_/T_num_edge_0.pickle')
# #         T_Label_1_num_edges = load_pickled_data(target_base_path + 'T_RUN_/T_num_edge_1.pickle')
# #     else:
# #         T_X_train_in = load_pickled_data(target_base_path + 'X_train_Label_1.pickle')
# #         T_y_train_in = load_pickled_data(target_base_path + 'y_train_Label_1.pickle')
# #         T_X_train_out = load_pickled_data(target_base_path + 'X_train_Label_0.pickle')
# #         T_y_train_out = load_pickled_data(target_base_path + 'y_train_Label_0.pickle')

# #     # print("T_X_train_in Size:{} and T_X_train_out Size:{}".format(len(T_X_train_in), len(T_X_train_out)))
# #     # Prepare Dataset
# #     X_attack = torch.FloatTensor(np.concatenate((S_X_train_in, S_X_train_out), axis=0))
# #     X_attack_nodes = torch.FloatTensor(np.concatenate((S_Label_1_num_nodes, S_Label_0_num_nodes), axis=0))
# #     X_attack_edges = torch.FloatTensor(np.concatenate((S_Label_1_num_edges, S_Label_0_num_edges), axis=0))

# #     y_target = torch.FloatTensor(np.concatenate((T_y_train_in, T_y_train_out), axis=0))
# #     y_attack = torch.FloatTensor(np.concatenate((S_y_train_in, S_y_train_out), axis=0))
# #     X_target = torch.FloatTensor(np.concatenate((T_X_train_in, T_X_train_out), axis=0))
# #     X_target_nodes = torch.FloatTensor(np.concatenate((T_Label_1_num_nodes, T_Label_0_num_nodes), axis=0))
# #     X_target_edges = torch.FloatTensor(np.concatenate((T_Label_1_num_edges, T_Label_0_num_edges), axis=0))

# #     feature_nums = min(X_attack.shape[1],X_target.shape[1])
# #     # print("feature_nums:{}".format(feature_nums))
# #     selected_X_target = select_top_k(X_target, feature_nums)
# #     selected_X_attack = select_top_k(X_attack, feature_nums)

# #     # selected_X_attack, selected_X_target = X_attack,X_target
# #     n_in = selected_X_attack.shape[1]
# #     attack_model = MLP(in_size=n_in, out_size=1, hidden_1=64, hidden_2=64)
# #     criterion = torch.nn.BCEWithLogitsLoss()
# #     optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.0001)
# #     attack_data = trainData(selected_X_attack, y_attack)
# #     target_data = testData(selected_X_target)
# #     train_loader = DataLoader(dataset=attack_data, batch_size=64, shuffle=True)
# #     target_loader = DataLoader(dataset=target_data, batch_size=1)
# #     all_acc = []
# #     for i in range(epochs):
# #         epoch_loss = 0
# #         epoch_acc = 0
# #         for X_batch, y_batch in train_loader:
# #             optimizer.zero_grad()
# #             y_pred = attack_model(X_batch)
# #             loss = criterion(y_pred, y_batch.unsqueeze(1))
# #             acc = binary_acc(y_pred, y_batch.unsqueeze(1))
# #             loss.backward()
# #             optimizer.step()

# #             epoch_loss += loss.item()
# #             epoch_acc += acc.item()
# #         all_acc.append(epoch_acc)
# #     y_pred_list = []
# #     attack_model.eval()
# #     correct_node_list, correct_edge_list = [],[]
# #     incorrect_node_list, incorrect_edge_list = [],[]
# #     with torch.no_grad():
# #         for X_batch, num_node, num_edge,y in zip(target_loader, X_target_nodes, X_target_edges, y_target):
# #             y_test_pred = attack_model(X_batch)
# #             y_test_pred = torch.sigmoid(y_test_pred)
# #             y_pred_tag = torch.round(y_test_pred)
# #             if y == y_pred_tag.detach().item():
# #                 correct_node_list.append(num_node.detach().item())
# #                 correct_edge_list.append(num_edge.detach().item())
# #             else:
# #                 incorrect_node_list.append(num_node.detach().item())
# #                 incorrect_edge_list.append(num_edge.detach().item())
# #             y_pred_list.append(y_pred_tag.cpu().numpy()[0])
# #     y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
# #     report = classification_report(y_target, y_pred_list)
# #     precision, recall, fscore, support = precision_recall_fscore_support(y_target,
# #                                                                          y_pred_list, average='macro')
# #     print(precision, recall)

# #     # print("correct_node_list:",correct_node_list)
# #     # print("correct_edge_list:",correct_edge_list)
# #     # print("incorrect_node_list:",incorrect_node_list)
# #     # print("incorrect_edge_list:",incorrect_edge_list)

# #     print(np.mean(correct_node_list), np.mean(correct_edge_list))
# #     print(np.mean(incorrect_node_list), np.mean(incorrect_edge_list))


# # if __name__ == '__main__':
# #     transfer_based_attack(300)

# import numpy as np
# import torch
# import random
# import glob
# from sklearn.model_selection import train_test_split
# from torch.utils.data import DataLoader
# from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score, f1_score, roc_auc_score
# import warnings
# import os
# from attack_models import MLP
# from utils import load_pickled_data, select_top_k, binary_acc, testData, trainData
# warnings.simplefilter("ignore")

# # def get_latest_run(base_path):
# #     runs = glob.glob(os.path.join(base_path, 'GCN_DD_GPU*'))
# #     return max(runs, key=os.path.getctime) if runs else None

# def get_latest_run(base_path):
#     runs = glob.glob(os.path.join(base_path, 'GatedGCN_CIFAR10_GPU*'))
#     return max(runs, key=os.path.getctime) if runs else None

# def load_data(run_path, run_type):
#     X_train_in = load_pickled_data(os.path.join(run_path, f'{run_type}_X_train_Label_1.pickle'))
#     y_train_in = load_pickled_data(os.path.join(run_path, f'{run_type}_y_train_Label_1.pickle'))
#     X_train_out = load_pickled_data(os.path.join(run_path, f'{run_type}_X_train_Label_0.pickle'))
#     y_train_out = load_pickled_data(os.path.join(run_path, f'{run_type}_y_train_Label_0.pickle'))
#     num_node_1 = load_pickled_data(os.path.join(run_path, f'{run_type}_num_node_1.pickle'))
#     num_node_0 = load_pickled_data(os.path.join(run_path, f'{run_type}_num_node_0.pickle'))
#     num_edge_1 = load_pickled_data(os.path.join(run_path, f'{run_type}_num_edge_1.pickle'))
#     num_edge_0 = load_pickled_data(os.path.join(run_path, f'{run_type}_num_edge_0.pickle'))
#     return X_train_in, y_train_in, X_train_out, y_train_out, num_node_1, num_node_0, num_edge_1, num_edge_0

# def transfer_based_attack(epochs):
#     # base_path = '/home/kzhao/MIA-GNN/results/TUs_graph_classification/checkpoints'
#     base_path = '/home/kzhao/MIA-GNN/out/SPs_graph_classification/checkpoints'
#     latest_run = get_latest_run(base_path)
#     if not latest_run:
#         raise FileNotFoundError(f"No data found in {base_path}")

#     print(f"Using data from: {latest_run}")

#     # Load shadow model data
#     S_X_train_in, S_y_train_in, S_X_train_out, S_y_train_out, S_Label_1_num_nodes, S_Label_0_num_nodes, S_Label_1_num_edges, S_Label_0_num_edges = load_data(os.path.join(latest_run, 'S_RUN_'), 'S')

#     # Load target model data
#     T_X_train_in, T_y_train_in, T_X_train_out, T_y_train_out, T_Label_1_num_nodes, T_Label_0_num_nodes, T_Label_1_num_edges, T_Label_0_num_edges = load_data(os.path.join(latest_run, 'T_RUN_'), 'T')

#     # Prepare Dataset
#     X_attack = torch.FloatTensor(np.concatenate((S_X_train_in, S_X_train_out), axis=0))
#     X_attack_nodes = torch.FloatTensor(np.concatenate((S_Label_1_num_nodes, S_Label_0_num_nodes), axis=0))
#     X_attack_edges = torch.FloatTensor(np.concatenate((S_Label_1_num_edges, S_Label_0_num_edges), axis=0))

#     y_target = torch.FloatTensor(np.concatenate((T_y_train_in, T_y_train_out), axis=0))
#     y_attack = torch.FloatTensor(np.concatenate((S_y_train_in, S_y_train_out), axis=0))
#     X_target = torch.FloatTensor(np.concatenate((T_X_train_in, T_X_train_out), axis=0))
#     X_target_nodes = torch.FloatTensor(np.concatenate((T_Label_1_num_nodes, T_Label_0_num_nodes), axis=0))
#     X_target_edges = torch.FloatTensor(np.concatenate((T_Label_1_num_edges, T_Label_0_num_edges), axis=0))

#     feature_nums = min(X_attack.shape[1], X_target.shape[1])
#     selected_X_target = select_top_k(X_target, feature_nums)
#     selected_X_attack = select_top_k(X_attack, feature_nums)

#     n_in = selected_X_attack.shape[1]
#     attack_model = MLP(in_size=n_in, out_size=1, hidden_1=64, hidden_2=64)
#     criterion = torch.nn.BCEWithLogitsLoss()
#     optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.0001)
#     attack_data = trainData(selected_X_attack, y_attack)
#     target_data = testData(selected_X_target)
#     train_loader = DataLoader(dataset=attack_data, batch_size=64, shuffle=True)
#     target_loader = DataLoader(dataset=target_data, batch_size=1)
#     all_acc = []
#     for i in range(epochs):
#         epoch_loss = 0
#         epoch_acc = 0
#         for X_batch, y_batch in train_loader:
#             optimizer.zero_grad()
#             y_pred = attack_model(X_batch)
#             loss = criterion(y_pred, y_batch.unsqueeze(1))
#             acc = binary_acc(y_pred, y_batch.unsqueeze(1))
#             loss.backward()
#             optimizer.step()

#             epoch_loss += loss.item()
#             epoch_acc += acc.item()
#         all_acc.append(epoch_acc)
#     y_pred_list = []
#     y_pred_prob_list = []  # 用于存储预测概率
#     attack_model.eval()
#     correct_node_list, correct_edge_list = [],[]
#     incorrect_node_list, incorrect_edge_list = [],[]
#     with torch.no_grad():
#         for X_batch, num_node, num_edge, y in zip(target_loader, X_target_nodes, X_target_edges, y_target):
#             y_test_pred = attack_model(X_batch)
#             y_test_pred = torch.sigmoid(y_test_pred)
#             y_pred_prob_list.append(y_test_pred.item())  # 新增：存储预测概率
#             y_pred_tag = torch.round(y_test_pred)
#             if y == y_pred_tag.detach().item():
#                 correct_node_list.append(num_node.detach().item())
#                 correct_edge_list.append(num_edge.detach().item())
#             else:
#                 incorrect_node_list.append(num_node.detach().item())
#                 incorrect_edge_list.append(num_edge.detach().item())
#             y_pred_list.append(y_pred_tag.cpu().numpy()[0])
#     y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
#     # report = classification_report(y_target, y_pred_list)
#     # precision, recall, fscore, support = precision_recall_fscore_support(y_target,
#     #                                                                      y_pred_list, average='macro')
#     # print(precision, recall)

#     # 计算各项指标
#     accuracy = accuracy_score(y_target, y_pred_list)
#     precision, recall, f1, _ = precision_recall_fscore_support(y_target, y_pred_list, average='macro')
#     auc = roc_auc_score(y_target, y_pred_prob_list)

#     # 打印结果
#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"F1 Score: {f1:.4f}")
#     print(f"AUC: {auc:.4f}")

#     print(f"Average nodes in correct predictions: {np.mean(correct_node_list):.2f}")
#     print(f"Average edges in correct predictions: {np.mean(correct_edge_list):.2f}")
#     print(f"Average nodes in incorrect predictions: {np.mean(incorrect_node_list):.2f}")
#     print(f"Average edges in incorrect predictions: {np.mean(incorrect_edge_list):.2f}")
#     # print(np.mean(correct_node_list), np.mean(correct_edge_list))
#     # print(np.mean(incorrect_node_list), np.mean(incorrect_edge_list))

# if __name__ == '__main__':
#     transfer_based_attack(300)

# # import numpy as np
# # import torch
# # import sys
# # import os
# # import torch.nn.functional as F
# # from dgl.data import TUDataset
# # from dgl.dataloading import GraphDataLoader
# # sys.path.append('/home/kzhao/MIA-GNN/code')
# # from nets.TUs_graph_classification.load_net import gnn_model
# # from tqdm import tqdm
# # import sys
# # import random
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import classification_report, precision_recall_fscore_support
# # import warnings
# # from attack_models import MLP
# # from utils import load_pickled_data, select_top_k, binary_acc, testData, trainData

# # # 添加项目根目录到 Python 路径
# # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # warnings.simplefilter("ignore")

# # def load_data(run_path, run_type):
# #     X_train_in = load_pickled_data(os.path.join(run_path, f'{run_type}_RUN_X_train_Label_1.pickle'))
# #     y_train_in = load_pickled_data(os.path.join(run_path, f'{run_type}_RUN_y_train_Label_1.pickle'))
# #     X_train_out = load_pickled_data(os.path.join(run_path, f'{run_type}_RUN_X_train_Label_0.pickle'))
# #     y_train_out = load_pickled_data(os.path.join(run_path, f'{run_type}_RUN_y_train_Label_0.pickle'))
# #     num_node_1 = load_pickled_data(os.path.join(run_path, f'{run_type}_RUN_num_node_1.pickle'))
# #     num_node_0 = load_pickled_data(os.path.join(run_path, f'{run_type}_RUN_num_node_0.pickle'))
# #     num_edge_1 = load_pickled_data(os.path.join(run_path, f'{run_type}_RUN_num_edge_1.pickle'))
# #     num_edge_0 = load_pickled_data(os.path.join(run_path, f'{run_type}_RUN_num_edge_0.pickle'))
# #     return X_train_in, y_train_in, X_train_out, y_train_out, num_node_1, num_node_0, num_edge_1, num_edge_0

# # def transfer_based_attack(epochs):
# #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
# #     # 加载DD数据集
# #     dataset = TUDataset(name='DD')
    
# #         # 获取图的特征维度
# #     graph, _ = dataset[0]
# #     if 'feat' in graph.ndata:
# #         in_dim = graph.ndata['feat'].shape[1]
# #     elif 'attr' in graph.ndata:
# #         in_dim = graph.ndata['attr'].shape[1]
# #     else:
# #         in_dim = 1  # 如果没有节点特征，使用度作为特征
    
# #     # 分割DD数据集为目标和影子部分
# #     num_graphs = len(dataset)
# #     indices = list(range(num_graphs))
# #     random.shuffle(indices)
# #     split = num_graphs // 2
# #     target_indices = indices[:split]
# #     shadow_indices = indices[split:]
    
# #     target_dataset = torch.utils.data.Subset(dataset, target_indices)
# #     shadow_dataset = torch.utils.data.Subset(dataset, shadow_indices)
    
# #     # 创建数据加载器
# #     target_loader = GraphDataLoader(target_dataset, batch_size=32, shuffle=True)
# #     shadow_loader = GraphDataLoader(shadow_dataset, batch_size=32, shuffle=True)
    
# #     # 定义模型参数
# #     MODEL_NAME = 'GCN'  # 或其他模型名称
# #     net_params = {
# #         'in_dim': in_dim,
# #         'hidden_dim': 64,
# #         'out_dim': dataset.num_classes,
# #         'n_classes': dataset.num_classes,
# #         'in_feat_dropout': 0.0,
# #         'dropout': 0.0,
# #         'L': 4,
# #         'readout': 'mean',
# #         'graph_norm': True,
# #         'batch_norm': True,
# #         'residual': True,
# #         'device': device
# #     }
    
# #     # 加载预训练模型
# #     model = gnn_model(MODEL_NAME, net_params)
# #     model.load_state_dict(torch.load('best_pretrained_model.pth'))
# #     model = model.to(device)
    
# #     # 微调目标模型和影子模型
# #     from finetune import finetune
# #     target_model = finetune(model, device, target_loader)
# #     shadow_model = finetune(model, device, shadow_loader)

# #     # 使用微调后的模型生成特征
# #     def generate_features(model, loader):
# #         features = []
# #         labels = []
# #         model.eval()
# #         with torch.no_grad():
# #             for batch_graphs, batch_labels in loader:
# #                 batch_graphs = batch_graphs.to(device)
# #                 batch_labels = batch_labels.to(device)
                
# #                 if 'feat' in batch_graphs.ndata:
# #                     batch_x = batch_graphs.ndata['feat'].float().to(device)
# #                 elif 'node_labels' in batch_graphs.ndata:
# #                     batch_x = batch_graphs.ndata['node_labels'].float().to(device)
# #                 else:
# #                     batch_x = torch.ones((batch_graphs.number_of_nodes(), 1), device=device)
                
# #                 out = model(batch_graphs, batch_x, None)
# #                 features.append(out.cpu().numpy())
# #                 labels.append(batch_labels.cpu().numpy())
# #         return np.concatenate(features), np.concatenate(labels)

# #     S_X, S_y = generate_features(shadow_model, shadow_loader)
# #     T_X, T_y = generate_features(target_model, target_loader)

# #     # 准备攻击数据
# #     X_attack = torch.FloatTensor(S_X)
# #     y_attack = torch.FloatTensor(S_y)
# #     X_target = torch.FloatTensor(T_X)
# #     y_target = torch.FloatTensor(T_y)

# #     feature_nums = min(X_attack.shape[1], X_target.shape[1])
# #     selected_X_target = select_top_k(X_target, feature_nums)
# #     selected_X_attack = select_top_k(X_attack, feature_nums)

# #     n_in = selected_X_attack.shape[1]
# #     attack_model = MLP(in_size=n_in, out_size=1, hidden_1=64, hidden_2=64)
# #     criterion = torch.nn.BCEWithLogitsLoss()
# #     optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.0001)
# #     attack_data = trainData(selected_X_attack, y_attack)
# #     target_data = testData(selected_X_target)
# #     train_loader = torch.utils.data.DataLoader(dataset=attack_data, batch_size=64, shuffle=True)
# #     target_loader = torch.utils.data.DataLoader(dataset=target_data, batch_size=1)
    
# #     # 训练攻击模型
# #     for i in range(epochs):
# #         epoch_loss = 0
# #         epoch_acc = 0
# #         for X_batch, y_batch in train_loader:
# #             optimizer.zero_grad()
# #             y_pred = attack_model(X_batch)
# #             loss = criterion(y_pred, y_batch.unsqueeze(1))
# #             acc = binary_acc(y_pred, y_batch.unsqueeze(1))
# #             loss.backward()
# #             optimizer.step()

# #             epoch_loss += loss.item()
# #             epoch_acc += acc.item()
# #         print(f'Epoch {i+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}, Acc: {epoch_acc/len(train_loader):.4f}')

# #     # 评估攻击模型
# #     y_pred_list = []
# #     attack_model.eval()
# #     with torch.no_grad():
# #         for X_batch in target_loader:
# #             y_test_pred = attack_model(X_batch)
# #             y_test_pred = torch.sigmoid(y_test_pred)
# #             y_pred_tag = torch.round(y_test_pred)
# #             y_pred_list.append(y_pred_tag.cpu().numpy()[0])
# #     y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    
# #     report = classification_report(y_target, y_pred_list)
# #     precision, recall, fscore, support = precision_recall_fscore_support(y_target, y_pred_list, average='macro')
# #     print("Classification Report:")
# #     print(report)
# #     print(f"Precision: {precision}, Recall: {recall}")

# # if __name__ == '__main__':
# #     transfer_based_attack(300)

import numpy as np
import torch
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, precision_recall_fscore_support
import warnings
import os
from attack_models import MLP
from utils import load_pickled_data, select_top_k, binary_acc, testData, trainData
warnings.simplefilter("ignore")


def transfer_based_attack(epochs):
    # GCN_DD_GPU1_12h53m37s_on_Jan_28_2021 0.7900280269	0.6378787879
    # GCN_DD_GPU0_19h36m32s_on_Jan_27_2021  0.822117084 0.7315151515

    # GCN_PROTEINS_full_GPU0_03h11m51s_on_Jan_28_2021 0.7707677769	0.5766666667
    attack_base_path = '/home/kzhao/mia_gnn/out/SPs_graph_classification/checkpoints/GatedGCN_CIFAR10_GPU0_12h10m41s_on_Aug_29_2024/'
    target_base_path = '/home/kzhao/mia_gnn/out/SPs_graph_classification/checkpoints/GatedGCN_CIFAR10_GPU0_12h10m41s_on_Aug_29_2024/'
    # GCN_ENZYMES_GPU0_16h40m29s_on_Jun_08_2021 -> GCN_DD_GPU1_16h26m32s_on_Jun_08_2021
    # For attack dataset
    if os.listdir(attack_base_path).__contains__("S_RUN_"):
        S_X_train_in = load_pickled_data(attack_base_path + 'S_RUN_/S_X_train_Label_1.pickle')
        S_y_train_in = load_pickled_data(attack_base_path + 'S_RUN_/S_y_train_Label_1.pickle')
        S_X_train_out = load_pickled_data(attack_base_path + 'S_RUN_/S_X_train_Label_0.pickle')
        S_y_train_out = load_pickled_data(attack_base_path + 'S_RUN_/S_y_train_Label_0.pickle')
        S_Label_0_num_nodes = load_pickled_data(attack_base_path + 'S_RUN_/S_num_node_0.pickle')
        S_Label_1_num_nodes = load_pickled_data(attack_base_path + 'S_RUN_/S_num_node_1.pickle')
        S_Label_0_num_edges = load_pickled_data(attack_base_path + 'S_RUN_/S_num_edge_0.pickle')
        S_Label_1_num_edges = load_pickled_data(attack_base_path + 'S_RUN_/S_num_edge_1.pickle')
    else:
        S_X_train_in = load_pickled_data(attack_base_path + 'X_train_Label_1.pickle')
        S_y_train_in = load_pickled_data(attack_base_path + 'y_train_Label_1.pickle')
        S_X_train_out = load_pickled_data(attack_base_path + 'X_train_Label_0.pickle')
        S_y_train_out = load_pickled_data(attack_base_path + 'y_train_Label_0.pickle')
    # For target Dataset
    if os.listdir(target_base_path).__contains__("T_RUN_"):
        T_X_train_in = load_pickled_data(target_base_path + 'T_RUN_/T_X_train_Label_1.pickle')
        T_y_train_in = load_pickled_data(target_base_path + 'T_RUN_/T_y_train_Label_1.pickle')
        T_X_train_out = load_pickled_data(target_base_path + 'T_RUN_/T_X_train_Label_0.pickle')
        T_y_train_out = load_pickled_data(target_base_path + 'T_RUN_/T_y_train_Label_0.pickle')
        T_Label_0_num_nodes = load_pickled_data(target_base_path + 'T_RUN_/T_num_node_0.pickle')
        T_Label_1_num_nodes = load_pickled_data(target_base_path + 'T_RUN_/T_num_node_1.pickle')
        T_Label_0_num_edges = load_pickled_data(target_base_path + 'T_RUN_/T_num_edge_0.pickle')
        T_Label_1_num_edges = load_pickled_data(target_base_path + 'T_RUN_/T_num_edge_1.pickle')
    else:
        T_X_train_in = load_pickled_data(target_base_path + 'X_train_Label_1.pickle')
        T_y_train_in = load_pickled_data(target_base_path + 'y_train_Label_1.pickle')
        T_X_train_out = load_pickled_data(target_base_path + 'X_train_Label_0.pickle')
        T_y_train_out = load_pickled_data(target_base_path + 'y_train_Label_0.pickle')

    # print("T_X_train_in Size:{} and T_X_train_out Size:{}".format(len(T_X_train_in), len(T_X_train_out)))
    # Prepare Dataset
    X_attack = torch.FloatTensor(np.concatenate((S_X_train_in, S_X_train_out), axis=0))
    X_attack_nodes = torch.FloatTensor(np.concatenate((S_Label_1_num_nodes, S_Label_0_num_nodes), axis=0))
    X_attack_edges = torch.FloatTensor(np.concatenate((S_Label_1_num_edges, S_Label_0_num_edges), axis=0))

    y_target = torch.FloatTensor(np.concatenate((T_y_train_in, T_y_train_out), axis=0))
    y_attack = torch.FloatTensor(np.concatenate((S_y_train_in, S_y_train_out), axis=0))
    X_target = torch.FloatTensor(np.concatenate((T_X_train_in, T_X_train_out), axis=0))
    X_target_nodes = torch.FloatTensor(np.concatenate((T_Label_1_num_nodes, T_Label_0_num_nodes), axis=0))
    X_target_edges = torch.FloatTensor(np.concatenate((T_Label_1_num_edges, T_Label_0_num_edges), axis=0))

    feature_nums = min(X_attack.shape[1],X_target.shape[1])
    # print("feature_nums:{}".format(feature_nums))
    selected_X_target = select_top_k(X_target, feature_nums)
    selected_X_attack = select_top_k(X_attack, feature_nums)

    # selected_X_attack, selected_X_target = X_attack,X_target
    n_in = selected_X_attack.shape[1]
    attack_model = MLP(in_size=n_in, out_size=1, hidden_1=64, hidden_2=64)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.0001)
    attack_data = trainData(selected_X_attack, y_attack)
    target_data = testData(selected_X_target)
    train_loader = DataLoader(dataset=attack_data, batch_size=64, shuffle=True)
    target_loader = DataLoader(dataset=target_data, batch_size=1)
    all_acc = []
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
        all_acc.append(epoch_acc)
    y_pred_list = []
    attack_model.eval()
    correct_node_list, correct_edge_list = [],[]
    incorrect_node_list, incorrect_edge_list = [],[]
    with torch.no_grad():
        for X_batch, num_node, num_edge,y in zip(target_loader, X_target_nodes, X_target_edges, y_target):
            y_test_pred = attack_model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            if y == y_pred_tag.detach().item():
                correct_node_list.append(num_node.detach().item())
                correct_edge_list.append(num_edge.detach().item())
            else:
                incorrect_node_list.append(num_node.detach().item())
                incorrect_edge_list.append(num_edge.detach().item())
            y_pred_list.append(y_pred_tag.cpu().numpy()[0])
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    report = classification_report(y_target, y_pred_list)
    precision, recall, fscore, support = precision_recall_fscore_support(y_target,
                                                                         y_pred_list, average='macro')
    print(precision, recall)

    # print("correct_node_list:",correct_node_list)
    # print("correct_edge_list:",correct_edge_list)
    # print("incorrect_node_list:",incorrect_node_list)
    # print("incorrect_edge_list:",incorrect_edge_list)

    print(np.mean(correct_node_list), np.mean(correct_edge_list))
    print(np.mean(incorrect_node_list), np.mean(incorrect_edge_list))


if __name__ == '__main__':
    transfer_based_attack(300)