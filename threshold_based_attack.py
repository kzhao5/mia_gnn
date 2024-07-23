# import os
# import pickle
# import torch
# from scipy.spatial import distance
# import numpy as np
# from torch import nn
# import pandas as pd
# from sklearn.metrics import accuracy_score

# def load_pickled_data(path):
#     with open(path, 'rb') as f:
#         unPickler = pickle.load(f)
#         return unPickler


# def load_data(m_path, nm_path):
#     data_in = load_pickled_data(m_path)
#     data_out = load_pickled_data(nm_path)
#     return data_in, data_out


# def softmax(x):
#     return np.exp(x) / sum(np.exp(x))


# def cal_distances(data):
#     # print('calculate distances...')
#     distance_matrix = []
#     for raw in data:
#         label = np.argmax(raw)
#         # cosine_dis = distance.cosine(label, raw)
#         euclid_dis = distance.euclidean(label, raw)
#         # corr_dis = distance.correlation(np.argmax(raw), raw) # nan
#         cheby_dis = distance.chebyshev(label, raw)
#         bray_dis = distance.braycurtis(label, raw)
#         canber_dis = distance.canberra(label, raw)
#         mahal_dis = distance.cityblock(label, raw)
#         sqeuclid_dis = distance.sqeuclidean(label, raw)
#         v = [euclid_dis, cheby_dis, bray_dis, canber_dis, mahal_dis, sqeuclid_dis]
#         distance_matrix.append(v)
#     return distance_matrix


# def cal_distance(data):
#     label = np.argmax(data)
#     # cosine_dis = distance.cosine(label, raw)
#     euclid_dis = distance.euclidean(label, data)
#     # corr_dis = distance.correlation(np.argmax(raw), raw) # nan
#     cheby_dis = distance.chebyshev(label, data)
#     bray_dis = distance.braycurtis(label, data)
#     canber_dis = distance.canberra(label, data)
#     mahal_dis = distance.cityblock(label, data)
#     sqeuclid_dis = distance.sqeuclidean(label, data)
#     v = [euclid_dis, cheby_dis, bray_dis, canber_dis, mahal_dis, sqeuclid_dis]
#     return v


# # Setup a plot such that only the bottom spine is shown
# def get_all_probabilities(data, factor):
#     return_list = []
#     for d in data:
#         return_list.append(d[np.argmax(d)] * factor)
#     return return_list


# def binary_acc(y_pred, y_test):
#     y_pred_tag = torch.round(torch.sigmoid(y_pred))

#     correct_results_sum = (y_pred_tag == y_test).sum().float()
#     acc = correct_results_sum / y_test.shape[0]
#     acc = torch.round(acc * 100)

#     return acc


# def get_all_exps(path):
#     dataset_list = ['CIFAR10', 'MNIST', 'DD', 'ENZYMES', 'PROTEINS_full', 'OGBG']
#     folders = os.listdir(path)
#     assert len(folders) > 0, "No dataset folder exist!"
#     exp_path_list = []
#     for folder in folders:
#         if dataset_list.__contains__(folder):
#             dataset_folder = os.path.join(path, folder)
#             exps = os.listdir(dataset_folder)
#             for exp in exps:
#                 exp_path = os.path.join(dataset_folder, exp)
#                 exp_path_list.append(exp_path)
#     return exp_path_list


# if __name__ == '__main__':
#     base_path = 'data/statis/GCN'
#     folders = os.listdir(base_path)
#     # print(sorted(folders))
#     for folder in folders:
#         all_exps = os.listdir(base_path + '/' + folder)
#         print(all_exps)
#         # for exp_path in all_exps:
#         #     exp_path = os.path.join(base_path + '/' + folder, exp_path)
#         # print(exp_path)
#         flag = 2
#         if all_exps.__contains__('S_RUN_'):
#             exp_path = os.path.join(base_path + '/' + folder, 'S_RUN_')
#             m_data_path = os.path.join(exp_path, 'S_X_train_Label_1.pickle')
#             nm_data_path = os.path.join(exp_path, 'S_X_train_Label_0.pickle')
#             S_Label_0_num_nodes = load_pickled_data(exp_path + '/S_num_node_0.pickle')
#             S_Label_1_num_nodes = load_pickled_data(exp_path + '/S_num_node_1.pickle')
#             S_Label_0_num_edges = load_pickled_data(exp_path + '/S_num_edge_0.pickle')
#             S_Label_1_num_edges = load_pickled_data(exp_path + '/S_num_edge_1.pickle')
#             X_attack_nodes = torch.FloatTensor(np.concatenate((S_Label_1_num_nodes, S_Label_0_num_nodes), axis=0))
#             X_attack_edges = torch.FloatTensor(np.concatenate((S_Label_1_num_edges, S_Label_0_num_edges), axis=0))
#             data_in, data_out = load_data(m_data_path, nm_data_path)
#             # LOSS function based attack
#             # ce_criterion = nn.CrossEntropyLoss()
#             # nl_criterion = nn.NLLLoss()
#             if flag == 1:
#                 mse_criterion = nn.MSELoss()
#                 ce_criterion = nn.CrossEntropyLoss()
#                 mse_in_loss_list, mse_out_loss_list, loss_diff_list = [], [], []
#                 ce_in_loss_list, ce_out_loss_list, ce_loss_diff_list = [], [], []
#                 with open('out/dd_loss_difference_calculation_single_instance.txt', 'a+') as writer:
#                     writer.write("For Experiment:{} \n".format(exp_path))
#                     for i in range(min(len(data_in), len(data_out))):
#                         x_in, x_in_label = data_in[i], np.argmax(data_in[i])
#                         x_out, x_out_label = data_out[i], np.argmax(data_out[i])

#                         ce_in_loss = ce_criterion(torch.FloatTensor([x_in]), torch.LongTensor([x_in_label]))
#                         ce_out_loss = ce_criterion(torch.FloatTensor([x_out]), torch.LongTensor([x_out_label]))

#                         mse_in_loss = mse_criterion(torch.FloatTensor([x_in]), torch.LongTensor([x_in_label]))
#                         mse_out_loss = mse_criterion(torch.FloatTensor([x_out]), torch.LongTensor([x_out_label]))

#                         mse_in_loss_list.append(float(mse_in_loss.numpy()))
#                         mse_out_loss_list.append(float(mse_out_loss.numpy()))

#                         ce_in_loss_list.append(float(ce_in_loss.numpy()))
#                         ce_out_loss_list.append(float(ce_out_loss.numpy()))

#                     loss_diff_list.append(np.mean(mse_in_loss_list) - np.mean(mse_out_loss_list))
#                     # writer.write("MSE Difference:{}\n".format(loss_diff_list))
#                     print(np.mean(ce_in_loss_list), np.mean(ce_out_loss_list))
#                     writer.write(
#                         "\t\tMSELLoss for Member:\n\t{} and Non-Member：\n\t{}\n".format(ce_in_loss_list,
#                                                                                         ce_out_loss_list))
#             elif flag == 2:
#                 print('plot all probabilities of {}'.format(exp_path))
#                 data_in_list = get_all_probabilities(data_in, 1)
#                 data_out_list = get_all_probabilities(data_out, 1)
#                 count_m = 0
#                 count_nm = 0
#                 t = np.array(range(9900, 10000)) / 10000
#                 with open('out/threshold_based_attack_result.txt', 'a+') as writer:
#                     writer.write("For Experiment:{} \n".format(exp_path))
#                     for t_ in t:
#                         correct_node_list, correct_edge_list = [], []
#                         incorrect_node_list, incorrect_edge_list = [], []
#                         y_true, y_pred_list = [],[]
#                         num_nodes, num_edges = [], []
#                         a = [x for x in data_in_list if x > t_]
#                         b = [x for x in data_out_list if x > t_]
#                         s_data = np.concatenate((data_in, data_out), axis=0)
#                         s_data_label = np.concatenate(([1 for i in data_in], [0 for j in data_in]), axis=0)
#                         max_prob_list = []
#                         for i in range(min(len(s_data), len(s_data_label))):
#                             # x_in, x_in_label = data_in[i], np.argmax(data_in[i])
#                             # x_out, x_out_label = data_out[i], np.argmax(data_out[i])
#                             max_prob_list.append(np.max(s_data[i]))
#                             num_nodes.append(X_attack_nodes[i].detach().item())
#                             num_edges.append(X_attack_edges[i].detach().item())
#                             # print(np.max(x_in), x_in_label)
#                         print(max_prob_list)
#                         # print(len(s_data),len(s_data_label), len(X_attack_nodes))
#                         writecsv = pd.DataFrame(
#                             {'num_node': num_nodes, 'num_edge': num_edges, 'max_prob': max_prob_list, 'label': s_data_label})
#                         print(writecsv)
#                         writecsv.to_csv("data/statis/test_results_with_threshold_" + str(t_) + ".csv",
#                                         index=False)
#                         # if x_in > t_:
#                         #     correct_node_list.append(n)
#                         #     correct_edge_list.append(e)
#                         #     y_pred_list.append(0)
#                         # else:
#                         #     incorrect_node_list.append(n)
#                         #     incorrect_edge_list.append(e)
#                         #     y_pred_list.append(1)
#                         # y_true.append(np.argmax(d))

#                     # for c, d, n, e in zip(s_data,s_data_value, X_attack_nodes, X_attack_edges):
#                     #     num_nodes.append(n.detach().item())
#                     #     num_edges.append(e.detach().item())
#                     #     if c > t_:
#                     #         correct_node_list.append(n)
#                     #         correct_edge_list.append(e)
#                     #         y_pred_list.append(0)
#                     #     else:
#                     #         incorrect_node_list.append(n)
#                     #         incorrect_edge_list.append(e)
#                     #         y_pred_list.append(1)
#                     #     y_true.append(np.argmax(d))
#                     # y = [y_t.detach().item() for y_t in y_target]
#                     # writecsv = pd.DataFrame(
#                     #     {'num_node': num_nodes, 'num_edge': num_edges, 'label': y_true, 'predict': y_pred_list})
#                     # print(writecsv)
#                     # writecsv.to_csv("data/statis/test_results_with_threshold_" + str(t_) + ".csv",
#                     #                 index=False)
#                     # writer.write(
#                     #     '\t\t With threshold:{}, Percentage for member is:{}, and for non-member is:{}, '
#                     #     'and num of correct node mean:{}, and num of correct edges:{}, '
#                     #     'and num of incorrect node mean:{}, '
#                     #     'and num of incorrect edge mean:{}'.format(
#                     #         t_,
#                     #         len(
#                     #             a) / len(
#                     #             data_in_list),
#                     #         len(
#                     #             b) / len(
#                     #             data_out_list), np.mean(correct_node_list), np.mean(correct_edge_list),
#                     #     np.mean(incorrect_node_list), np.mean(incorrect_edge_list)))
#                     # writer.write('\n')
#                     #
#                     # print(accuracy_score(y_true,y_pred_list))




# import os
# import pickle
# import dgl
# import sys
# import glob
# import torch
# import torch.cuda
# import numpy as np
# from torch import nn
# import pandas as pd
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
# sys.path.append('/home/kzhao/MIA-GNN/code')
# from nets.TUs_graph_classification.load_net import gnn_model
# from dgl.data import TUDataset

# def load_pickled_data(path):
#     with open(path, 'rb') as f:
#         return pickle.load(f)

# def load_data(path, prefix):
#     X_train_in = load_pickled_data(os.path.join(path, f'{prefix}_X_train_Label_1.pickle'))
#     y_train_in = load_pickled_data(os.path.join(path, f'{prefix}_y_train_Label_1.pickle'))
#     X_train_out = load_pickled_data(os.path.join(path, f'{prefix}_X_train_Label_0.pickle'))
#     y_train_out = load_pickled_data(os.path.join(path, f'{prefix}_y_train_Label_0.pickle'))
#     Label_1_num_nodes = load_pickled_data(os.path.join(path, f'{prefix}_num_node_1.pickle'))
#     Label_0_num_nodes = load_pickled_data(os.path.join(path, f'{prefix}_num_node_0.pickle'))
#     Label_1_num_edges = load_pickled_data(os.path.join(path, f'{prefix}_num_edge_1.pickle'))
#     Label_0_num_edges = load_pickled_data(os.path.join(path, f'{prefix}_num_edge_0.pickle'))
#     return X_train_in, y_train_in, X_train_out, y_train_out, Label_1_num_nodes, Label_0_num_nodes, Label_1_num_edges, Label_0_num_edges

# def load_model(model_path, net_params, MODEL_NAME, device):
#     model = gnn_model(MODEL_NAME, net_params)
#     state_dict = torch.load(model_path, map_location=device)
    
#     new_state_dict = {}
#     for k, v in state_dict.items():
#         if k in model.state_dict():
#             if v.shape == model.state_dict()[k].shape:
#                 new_state_dict[k] = v.to(device)
#             else:
#                 print(f"Shape mismatch for {k}: saved {v.shape}, current {model.state_dict()[k].shape}")
#                 if k == 'embedding_h.weight' or k == 'embedding_h.bias':
#                     new_state_dict[k] = model.state_dict()[k].to(device)
#                 elif k == 'MLP_layer.FC_layers.2.weight' or k == 'MLP_layer.FC_layers.2.bias':
#                     new_state_dict[k] = model.state_dict()[k].to(device)
    
#     model.load_state_dict(new_state_dict, strict=False)
#     return model.to(device)

# def get_predictions(model, g, features, device):
#     model.eval()
#     probs_list = []
#     with torch.no_grad():
#         for i in range(features.shape[0]):
#             single_g = dgl.graph(([0], [0])).to(device)
#             single_g.ndata['feat'] = features[i].unsqueeze(0)
#             logits = model(single_g, single_g.ndata['feat'], None)
#             probs = torch.softmax(logits, dim=1)
#             probs_list.append(probs.cpu().numpy())
#     return np.concatenate(probs_list, axis=0)

# def threshold_based_attack(probs, labels, thresholds):
#     print(f"Shape of probs: {probs.shape}")
#     print(f"Shape of labels: {labels.shape}")
#     print(f"Number of thresholds: {len(thresholds)}")
    
#     results = []
#     for t in thresholds:
#         predictions = ((probs[:, 0] < t) & (probs[:, 1] > t)).astype(int)
#         print(f"Shape of predictions for threshold {t}: {predictions.shape}")
#         print(f"First few predictions: {predictions[:5]}")
#         print(f"First few labels: {labels[:5]}")
#         print(f"Prediction distribution: {np.bincount(predictions)}")
        
#         acc = accuracy_score(labels, predictions)
#         prec = precision_score(labels, predictions, zero_division=0)
#         rec = recall_score(labels, predictions, zero_division=0)
#         f1 = f1_score(labels, predictions, zero_division=0)
#         results.append((t, acc, prec, rec, f1))
#     return results

# def get_latest_run(base_path):
#     runs = glob.glob(os.path.join(base_path, 'GCN_DD_GPU*'))
#     return max(runs, key=os.path.getctime) if runs else None

# def main():
#     base_path = '/home/kzhao/MIA-GNN/results/TUs_graph_classification/checkpoints'
#     MODEL_NAME = 'GCN'
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")

#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.set_per_process_memory_fraction(0.8)

#     latest_run = get_latest_run(base_path)
#     if not latest_run:
#         raise FileNotFoundError(f"No data found in {base_path}")
    
#     exp_name = os.path.basename(latest_run)
#     print(f"Processing latest experiment: {exp_name}")
    
#     dataset = TUDataset(name='DD')
#     graph, _ = dataset[0]
    
#     T_X_train_in, T_y_train_in, T_X_train_out, T_y_train_out, T_Label_1_num_nodes, T_Label_0_num_nodes, T_Label_1_num_edges, T_Label_0_num_edges = load_data(os.path.join(latest_run, 'T_RUN_'), 'T')
    
#     features = torch.FloatTensor(np.concatenate((T_X_train_in, T_X_train_out), axis=0)).to(device)
#     labels = torch.LongTensor(np.concatenate((T_y_train_in, T_y_train_out), axis=0)).to(device)
    
#     net_params = {
#         'in_dim': features.shape[1],
#         'hidden_dim': 138,
#         'out_dim': 138,
#         'n_classes': dataset.num_classes,  
#         'in_feat_dropout': 0.0,
#         'dropout': 0.0,
#         'L': 4,
#         'readout': 'mean',
#         'graph_norm': True,
#         'batch_norm': True,
#         'residual': True,
#         'device': device
#     }
    
#     t_run_path = os.path.join(latest_run, 'T_RUN_')
#     model_path = os.path.join(t_run_path, 'target_model.pth')
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model file not found: {model_path}")
    
#     model = load_model(model_path, net_params, MODEL_NAME, device)
#     model = model.to(device)
#     for param in model.parameters():
#         param.requires_grad = False

#     num_nodes = features.shape[0]
#     g = dgl.graph((torch.arange(num_nodes), torch.arange(num_nodes))).to(device)
#     g.ndata['feat'] = features

#     model.embedding_h = nn.Linear(features.shape[1], net_params['hidden_dim']).to(device)
#     model.MLP_layer.FC_layers[-1] = nn.Linear(model.MLP_layer.FC_layers[-1].in_features, 2).to(device)

#     print(f"Features shape: {features.shape}")
#     print(f"Labels shape: {labels.shape}")
#     print(f"Number of nodes in graph: {g.number_of_nodes()}")
#     print(f"Number of edges in graph: {g.number_of_edges()}")

#     # 在 main 函数中
#     g = dgl.graph(([i for i in range(features.shape[0])], [i for i in range(features.shape[0])]))
#     g = g.to(device)
#     features = features.to(device)
#     g.ndata['feat'] = features
    

#     # Get predictions
#     probs = get_predictions(model, g, features, device)
#     n_train = len(T_X_train_in)
#     n_test = len(T_X_train_out)
#     true_labels = np.concatenate([np.ones(n_train), np.zeros(n_test)])


#     print(f"Shape of probs before attack: {probs.shape}")
#     print(f"Shape of true_labels: {true_labels.shape}")
#     print(f"First few probs: {probs[:5]}")
#     print(f"First few true_labels: {true_labels[:5]}")
#     # Perform threshold-based attack
#     thresholds = np.linspace(0.1, 0.9, 51)
#     results = threshold_based_attack(probs, true_labels, thresholds)
    
#     # Find best result
#     best_result = max(results, key=lambda x: x[4])  # x[4] is F1 score
#     print(f"Best result - Threshold: {best_result[0]:.2f}, Accuracy: {best_result[1]:.4f}, Precision: {best_result[2]:.4f}, Recall: {best_result[3]:.4f}, F1: {best_result[4]:.4f}")
#     # Save results
#     df = pd.DataFrame(results, columns=['Threshold', 'Accuracy', 'Precision', 'Recall', 'F1'])
#     df.to_csv(f'threshold_based_mia_results_{exp_name}.csv', index=False)

#     # Calculate AUC
#     fpr, tpr, _ = roc_curve(true_labels, np.max(probs, axis=1))
#     roc_auc = auc(fpr, tpr)
#     print(f"AUC: {roc_auc:.4f}")

#     # 打印一些样本的详细信息
#     print("\nDetailed sample information:")
#     for i in range(10):  # 打印前10个样本
#         print(f"Sample {i}: Prob[0]={probs[i,0]:.4f}, Prob[1]={probs[i,1]:.4f}, True Label={true_labels[i]}")

#     # 打印预测概率的统计信息
#     print("\nProbability statistics:")
#     print(f"Mean of Prob[0]: {np.mean(probs[:,0]):.4f}")
#     print(f"Mean of Prob[1]: {np.mean(probs[:,1]):.4f}")
#     print(f"Min of Prob[1]: {np.min(probs[:,1]):.4f}")
#     print(f"Max of Prob[1]: {np.max(probs[:,1]):.4f}")

# if __name__ == "__main__":
#     main()
import os
import pickle
import dgl
import sys
import glob
import torch
import torch.cuda
import numpy as np
from torch import nn
import pandas as pd
from scipy.spatial import distance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
sys.path.append('/home/kzhao/MIA-GNN/code')
from nets.TUs_graph_classification.load_net import gnn_model
from dgl.data import TUDataset

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

def get_predictions(model, graphs, device):
    model.eval()
    probs_list = []
    with torch.no_grad():
        for g in graphs:
            # Graph and its features should already be on the correct device
            logits = model(g, g.ndata['feat'], None)
            probs = torch.softmax(logits, dim=1)
            probs_list.append(probs.cpu().numpy())
    return np.concatenate(probs_list, axis=0)

def cal_distances(data):
    label = np.argmax(data)
    euclid_dis = distance.euclidean([label], data)
    cheby_dis = distance.chebyshev([label], data)
    bray_dis = distance.braycurtis([label], data)
    canber_dis = distance.canberra([label], data)
    mahal_dis = distance.cityblock([label], data)
    sqeuclid_dis = distance.sqeuclidean([label], data)
    return [euclid_dis, cheby_dis, bray_dis, canber_dis, mahal_dis, sqeuclid_dis]

def threshold_based_attack(probs, labels, thresholds):
    print(f"Shape of probs: {probs.shape}")
    print(f"Shape of labels: {labels.shape}")
    print(f"Number of thresholds: {len(thresholds)}")
    
    results = []
    for t in thresholds:
        predictions = ((probs[:, 0] < t) & (probs[:, 1] > t)).astype(int)
        print(f"Shape of predictions for threshold {t}: {predictions.shape}")
        print(f"First few predictions: {predictions[:5]}")
        print(f"First few labels: {labels[:5]}")
        print(f"Prediction distribution: {np.bincount(predictions)}")
        
        acc = accuracy_score(labels, predictions)
        prec = precision_score(labels, predictions, zero_division=0)
        rec = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        results.append((t, acc, prec, rec, f1))
    return results

def get_latest_run(base_path):
    runs = glob.glob(os.path.join(base_path, 'GCN_DD_GPU*'))
    return max(runs, key=os.path.getctime) if runs else None

def main():
    base_path = '/home/kzhao/MIA-GNN/results/TUs_graph_classification/checkpoints'
    MODEL_NAME = 'GCN'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.8)

    latest_run = get_latest_run(base_path)
    if not latest_run:
        raise FileNotFoundError(f"No data found in {base_path}")
    
    exp_name = os.path.basename(latest_run)
    print(f"Processing latest experiment: {exp_name}")
    
    dataset = TUDataset(name='DD')
    
    T_X_train_in, T_y_train_in, T_X_train_out, T_y_train_out, T_Label_1_num_nodes, T_Label_0_num_nodes, T_Label_1_num_edges, T_Label_0_num_edges = load_data(os.path.join(latest_run, 'T_RUN_'), 'T')
    
    features = torch.FloatTensor(np.concatenate((T_X_train_in, T_X_train_out), axis=0)).to(device)
    labels = torch.LongTensor(np.concatenate((T_y_train_in, T_y_train_out), axis=0)).to(device)
    num_nodes = torch.FloatTensor(np.concatenate((T_Label_1_num_nodes, T_Label_0_num_nodes), axis=0)).to(device)
    num_edges = torch.FloatTensor(np.concatenate((T_Label_1_num_edges, T_Label_0_num_edges), axis=0)).to(device)
    
    net_params = {
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
    
    t_run_path = os.path.join(latest_run, 'T_RUN_')
    model_path = os.path.join(t_run_path, 'target_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = load_model(model_path, net_params, MODEL_NAME, device)
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False

        # After loading the model
    print(f"Model's embedding_h.weight shape: {model.embedding_h.weight.shape}")
    print(f"Features shape: {features.shape}")

    if model.embedding_h.weight.shape[1] != features.shape[1]:
        print(f"Adjusting embedding layer to match input features dimension")
        model.embedding_h = nn.Linear(features.shape[1], net_params['hidden_dim']).to(device)

    # Make sure the final layer is also correct
    model.MLP_layer.FC_layers[-1] = nn.Linear(model.MLP_layer.FC_layers[-1].in_features, 2).to(device)

    # model.embedding_h = nn.Linear(features.shape[1], net_params['hidden_dim']).to(device)
    # model.MLP_layer.FC_layers[-1] = nn.Linear(model.MLP_layer.FC_layers[-1].in_features, 2).to(device)

    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Num nodes shape: {num_nodes.shape}")
    print(f"Num edges shape: {num_edges.shape}")

    # Create graph structures
    graphs = []
    for i in range(len(num_nodes)):
        num_nodes_i = int(num_nodes[i].item())
        num_edges_i = int(num_edges[i].item())
        
        # Create graph with correct number of nodes
        src = torch.randint(0, num_nodes_i, (num_edges_i,))
        dst = torch.randint(0, num_nodes_i, (num_edges_i,))
        g = dgl.graph((src, dst), num_nodes=num_nodes_i)
        g = g.to(device)
        # Set graph-level features to all nodes
        node_features = features[i].repeat(num_nodes_i, 1)
        g.ndata['feat'] = node_features

        
        graphs.append(g)

    # Get predictions
    probs = get_predictions(model, graphs, device)
    n_train = len(T_X_train_in)
    n_test = len(T_X_train_out)
    true_labels = np.concatenate([np.ones(n_train), np.zeros(n_test)])

    print(f"Shape of probs before attack: {probs.shape}")
    print(f"Shape of true_labels: {true_labels.shape}")
    print(f"First few probs: {probs[:5]}")
    print(f"First few true_labels: {true_labels[:5]}")



    # Calculate distances
    distances = np.array([cal_distances(prob) for prob in probs])

    # Perform threshold-based attack
    thresholds = np.linspace(0.1, 0.9, 51)
    results = threshold_based_attack(probs, true_labels, thresholds)
    
    # Find best result
    best_result = max(results, key=lambda x: x[4])  # x[4] is F1 score
    print(f"Best result - Threshold: {best_result[0]:.2f}, Accuracy: {best_result[1]:.4f}, Precision: {best_result[2]:.4f}, Recall: {best_result[3]:.4f}, F1: {best_result[4]:.4f}")
    
    # Save results
    df = pd.DataFrame(results, columns=['Threshold', 'Accuracy', 'Precision', 'Recall', 'F1'])
    df.to_csv(f'threshold_based_mia_results_{exp_name}.csv', index=False)

    # Calculate AUC
    fpr, tpr, _ = roc_curve(true_labels, np.max(probs, axis=1))
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc:.4f}")

    # 打印一些样本的详细信息
    print("\nDetailed sample information:")
    for i in range(10):  # 打印前10个样本
        print(f"Sample {i}: Prob[0]={probs[i,0]:.4f}, Prob[1]={probs[i,1]:.4f}, True Label={true_labels[i]}")
        print(f"Distances: {distances[i]}")

    # 打印预测概率的统计信息
    print("\nProbability statistics:")
    print(f"Mean of Prob[0]: {np.mean(probs[:,0]):.4f}")
    print(f"Mean of Prob[1]: {np.mean(probs[:,1]):.4f}")
    print(f"Min of Prob[1]: {np.min(probs[:,1]):.4f}")
    print(f"Max of Prob[1]: {np.max(probs[:,1]):.4f}")

    # 保存更多详细信息
    detailed_results = pd.DataFrame({
        'num_nodes': num_nodes.cpu().numpy(),
        'num_edges': num_edges.cpu().numpy(),
        'prob_0': probs[:, 0],
        'prob_1': probs[:, 1],
        'true_label': true_labels
    })
    for i, dist_name in enumerate(['euclid', 'cheby', 'bray', 'canber', 'mahal', 'sqeuclid']):
        detailed_results[f'{dist_name}_dis'] = distances[:, i]
    
    detailed_results.to_csv(f'detailed_mia_results_{exp_name}.csv', index=False)

if __name__ == "__main__":
    main()