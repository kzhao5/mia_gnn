
# """
#     IMPORTING LIBS
# """
# import dgl
# import numpy as np
# import os
# import socket
# import time
# import random
# import glob
# import argparse, json
# import pickle

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import torch.optim as optim
# from torch.utils.data import DataLoader

# from tensorboardX import SummaryWriter
# from torch.utils.data.dataset import random_split
# from tqdm import tqdm
# from train.train_SPs_graph_classification import train_epoch_sparse as train_epoch, evaluate_network_sparse as evaluate_network
# from dgl.data import CIFAR10SuperPixelDataset
# from dgl.data import MNISTSuperPixelDataset
# from torch.utils.data import Subset
# from scipy.stats import norm
# from tqdm import tqdm
# from torch.utils.data import ConcatDataset
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
# import matplotlib.pyplot as plt
# import seaborn as sns

# class DotDict(dict):
#     def __init__(self, **kwds):
#         self.update(kwds)
#         self.__dict__ = self






# """
#     IMPORTING CUSTOM MODULES/METHODS
# """
# from nets.SPs_graph_classification.load_net import gnn_model # import all GNNS
# from data.data import LoadData # import dataset



# """
#     GPU Setup
# """
# def gpu_setup(use_gpu, gpu_id):
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

#     if torch.cuda.is_available() and use_gpu:
#         print('cuda available with GPU:' ,torch.cuda.get_device_name(0))
#         device = torch.device("cuda")
#     else:
#         print('cuda not available')
#         device = torch.device("cpu")
#     return device










# """
#     VIEWING MODEL CONFIG AND PARAMS
# """
# def view_model_param(MODEL_NAME, net_params):
#     model = gnn_model(MODEL_NAME, net_params)
#     total_param = 0
#     print("MODEL DETAILS:\n")
#     # print(model)
#     for param in model.parameters():
#         # print(param.data.size())
#         total_param += np.prod(list(param.data.size()))
#     print('MODEL/Total parameters:', MODEL_NAME, total_param)
#     return total_param


# """
#     TRAINING CODE
# """

# def manipulate_model(MODEL_NAME, net_params, target_aux_dataset, params, device, initial_alpha=0.1, max_alpha=0.5, alpha_increase_rate=0.01, initial_lr=0.001, min_lr=0.0001, lr_decay_rate=0.95):
#     print("------------Start manipulate------------")

#     pretrain_model_path = 'SP_pretrain_model.pth'
#     # 加载预训练的模型
#     model_manipulate = gnn_model(MODEL_NAME, net_params)
#     model_manipulate.load_state_dict(torch.load(pretrain_model_path))
#     model_manipulate = model_manipulate.to(device)
#     manipulate_optimizer = optim.Adam(model_manipulate.parameters(), lr=initial_lr)
#     scheduler = optim.lr_scheduler.ExponentialLR(manipulate_optimizer, gamma=lr_decay_rate)


#     # 确保所有参数都需要梯度
#     for param in model_manipulate.parameters():
#         param.requires_grad = True

#     # 创建 Dtarget 和 Daux 的数据加载器
#     target_loader = DataLoader(target_aux_dataset.target, batch_size=params['batch_size'], shuffle=True, collate_fn=target_aux_dataset.collate)
#     aux_loader = DataLoader(target_aux_dataset.aux, batch_size=params['batch_size'], shuffle=True, collate_fn=target_aux_dataset.collate)

#     num_manipulate_epochs = 100
#     alpha = 0.5  # 控制poisoning强度的系数
#     # alpha = initial_alpha
#     best_diff = 0
#     patience = 10
#     no_improve = 0

#     for epoch in range(num_manipulate_epochs):
#         model_manipulate.train()
#         total_loss = 0
#         total_target_loss = 0
#         total_aux_loss = 0

#         # 处理 Dtarget
#         for batch_graphs, batch_labels in target_loader:
#             batch_graphs = batch_graphs.to(device)
#             batch_x = batch_graphs.ndata['feat'].to(device)
#             batch_e = batch_graphs.edata['feat'].to(device)
#             batch_labels = batch_labels.long().to(device)

#             out = model_manipulate(batch_graphs, batch_x, batch_e)
#             loss_target = F.cross_entropy(out, batch_labels)
            
#             total_target_loss += loss_target

#         # 处理 Daux
#         for batch_graphs, batch_labels in aux_loader:
#             batch_graphs = batch_graphs.to(device)
#             batch_x = batch_graphs.ndata['feat'].to(device)
#             batch_e = batch_graphs.edata['feat'].to(device)
#             batch_labels = batch_labels.long().to(device)

#             out = model_manipulate(batch_graphs, batch_x, batch_e)
#             loss_aux = F.cross_entropy(out, batch_labels)
            
#             total_aux_loss += loss_aux

#         # 计算总的损失
#         avg_target_loss = total_target_loss / len(target_loader)
#         avg_aux_loss = total_aux_loss / len(aux_loader)
#         # loss = alpha * avg_aux_loss - (1 - alpha) * avg_target_loss
#         loss = alpha * avg_aux_loss

#         manipulate_optimizer.zero_grad()
#         loss.backward()
#         manipulate_optimizer.step()

#         print(f'Manipulating Epoch {epoch+1}/{num_manipulate_epochs}, '
#               f'Loss: {loss.item():.4f}, Target Loss: {avg_target_loss.item():.4f}, Aux Loss: {avg_aux_loss.item():.4f}')

#     print("Manipulation completed. Saving model.")
#     torch.save(model_manipulate.state_dict(), 'manipulated_model.pth')
#     return model_manipulate

# def train_val_pipeline(device, MODEL_NAME, dataset, params, net_params, dirs, target_aux_dataset, pretrained_path):
#     t0 = time.time()
#     per_epoch_time = []
    
#     DATASET_NAME = dataset.name
#     # root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs

#     # 创建新的保存目录
#     current_file_path = os.path.abspath(__file__)
#     project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
#     save_dir = os.path.join(project_root, "lira_out")
#     os.makedirs(save_dir, exist_ok=True)
    
#     # setting seeds
#     random.seed(params['seed'])
#     np.random.seed(params['seed'])
#     torch.manual_seed(params['seed'])
#     if device.type == 'cuda':
#         torch.cuda.manual_seed(params['seed'])
    
#     print("Target Dataset: ", DATASET_NAME)

#     manipulated_model = manipulate_model(MODEL_NAME, net_params, target_aux_dataset, params, device)
    
#     def load_pretrained_model():
#         model = gnn_model(MODEL_NAME, net_params)
#         model.load_state_dict(torch.load(pretrained_path))
#         return model.to(device)

#     # Load and train Target Model
#     t_model = load_pretrained_model()
#     print(f"Loaded pre-trained model from {pretrained_path}")

#     # Train target model
#     target_train_loader = DataLoader(dataset.train, batch_size=params['batch_size'], shuffle=True, collate_fn=dataset.collate)
#     target_val_loader = DataLoader(dataset.val, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    
#     optimizer = optim.Adam(t_model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#                                                      factor=params['lr_reduce_factor'],
#                                                      patience=params['lr_schedule_patience'],
#                                                      verbose=True)
    
#     print("Training Target Model")
#     for epoch in range(params['epochs']):
#         t1 = time.time()
#         epoch_train_loss, epoch_train_acc, optimizer = train_epoch(t_model, optimizer, device, target_train_loader, epoch)
#         epoch_val_loss, epoch_val_acc = evaluate_network(t_model, device, target_val_loader, epoch)
        
#         scheduler.step(epoch_val_loss)
        
#         per_epoch_time.append(time.time()-t1)
        
#         print(f"Target Model: Epoch {epoch + 1}, train_loss: {epoch_train_loss:.4f}, val_loss: {epoch_val_loss:.4f}, val_acc: {epoch_val_acc:.4f}")
        
#         if optimizer.param_groups[0]['lr'] < params['min_lr']:
#             print("\n!! LR EQUAL TO MIN LR SET.")
#             break
    
#     # LiRA Attack Class
#     class LiRAAttack:
#         def __init__(self, shadow_models, n_queries=64):
#             self.shadow_models = shadow_models
#             self.n_queries = n_queries

#         def fit(self, shadow_loader, is_member):
#             self.member_scores = []
#             self.non_member_scores = []
            
#             for shadow_model in self.shadow_models:
#                 shadow_model.eval()
            
#             with torch.no_grad():
#                 for batch_graphs, batch_labels in tqdm(shadow_loader, desc="Fitting LiRA"):
#                     batch_graphs = batch_graphs.to(device)
#                     batch_labels = batch_labels.long().to(device)
                    
#                     batch_scores = []
                    
#                     for _ in range(self.n_queries):
#                         logits = [model(batch_graphs, batch_graphs.ndata['feat'], batch_graphs.edata['feat']) for model in self.shadow_models]
#                         scores = torch.stack([l[torch.arange(len(batch_labels)), batch_labels] for l in logits], dim=1)
#                         batch_scores.append(scores)
                    
#                     batch_scores = torch.stack(batch_scores, dim=1).mean(dim=1)
                    
#                     for score, is_mem in zip(batch_scores, is_member):
#                         if is_mem:
#                             self.member_scores.append(score.cpu().numpy())
#                         else:
#                             self.non_member_scores.append(score.cpu().numpy())
            
#             self.member_params = self._fit_gaussian(self.member_scores)
#             self.non_member_params = self._fit_gaussian(self.non_member_scores)

#         def predict(self, target_model, target_loader):
#             target_model.eval()
#             predictions = []
#             confidences = []
            
#             with torch.no_grad():
#                 for batch_graphs, batch_labels in tqdm(target_loader, desc="LiRA Prediction"):
#                     batch_graphs = batch_graphs.to(device)
#                     batch_labels = batch_labels.long().to(device)
                    
#                     batch_scores = []
                    
#                     for _ in range(self.n_queries):
#                         logits = target_model(batch_graphs, batch_graphs.ndata['feat'], batch_graphs.edata['feat'])
#                         scores = logits[torch.arange(len(batch_labels)), batch_labels]
#                         batch_scores.append(scores)
                    
#                     batch_scores = torch.stack(batch_scores, dim=1).mean(dim=1)
                    
#                     for score in batch_scores:
#                         member_likelihood = self._gaussian_likelihood(score.item(), *self.member_params)
#                         non_member_likelihood = self._gaussian_likelihood(score.item(), *self.non_member_params)
#                         # likelihood_ratio = member_likelihood / non_member_likelihood
#                         if non_member_likelihood == 0:
#                             likelihood_ratio = float('inf')  # 或者设置一个大数字
#                         else:
#                             likelihood_ratio = member_likelihood / non_member_likelihood
#                         predictions.append(likelihood_ratio > 1)
#                         confidences.append(likelihood_ratio)
            
#             return np.array(predictions), np.array(confidences)

#         def _fit_gaussian(self, data):
#             return np.mean(data), np.std(data)

#         def _gaussian_likelihood(self, x, mean, std):
#             return norm.pdf(x, mean, std)

#     # Train multiple shadow models
#     n_shadow_models = 10
#     shadow_models = []
    
#     # 使用shadow数据训练shadow models
#     shadow_train_loader = DataLoader(dataset.shadow_train, batch_size=params['batch_size'], shuffle=True, collate_fn=dataset.collate)
#     shadow_val_loader = DataLoader(dataset.shadow_val, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    
#     for i in range(n_shadow_models):
#         print(f"Training Shadow Model {i+1}/{n_shadow_models}")
#         # shadow_model = gnn_model(MODEL_NAME, net_params)
#         shadow_model = load_pretrained_model()  # Load pre-trained model for each shadow model
#         shadow_model = shadow_model.to(device)
#         optimizer = optim.Adam(shadow_model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#                                                          factor=params['lr_reduce_factor'],
#                                                          patience=params['lr_schedule_patience'],
#                                                          verbose=True)
        
#         # train shadow model
#         for epoch in range(params['epochs']):
#             t1 = time.time()
#             epoch_train_loss, epoch_train_acc, optimizer = train_epoch(shadow_model, optimizer, device, shadow_train_loader, epoch)
#             epoch_val_loss, epoch_val_acc = evaluate_network(shadow_model, device, shadow_val_loader, epoch)
            
#             scheduler.step(epoch_val_loss)
            
#             per_epoch_time.append(time.time()-t1)
            
#             print(f"Shadow Model {i+1}: Epoch {epoch + 1}, train_loss: {epoch_train_loss:.4f}, val_loss: {epoch_val_loss:.4f}, val_acc: {epoch_val_acc:.4f}")
            
#             if optimizer.param_groups[0]['lr'] < params['min_lr']:
#                 print("\n!! LR EQUAL TO MIN LR SET.")
#                 break
                
#         shadow_models.append(shadow_model)
    
#     print("=================Start LiRA Attack=================")
    
#     # Prepare shadow data
#     shadow_dataset = ConcatDataset([dataset.shadow_train, dataset.shadow_val])
#     shadow_loader = DataLoader(shadow_dataset, batch_size=params['batch_size'], shuffle=True, collate_fn=dataset.collate)
#     is_member = torch.cat([torch.ones(len(dataset.shadow_train)), torch.zeros(len(dataset.shadow_val))])

#     # Create and train LiRA attack
#     lira = LiRAAttack(shadow_models, n_queries=64)
#     lira.fit(shadow_loader, is_member)

#     # Attack target model
#     target_dataset = ConcatDataset([dataset.train, dataset.val])
#     target_loader = DataLoader(target_dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
#     predictions, confidences = lira.predict(t_model, target_loader)
#     confidences = np.array(confidences, dtype=np.float64)
#     confidences = np.nan_to_num(confidences, nan=0.0, posinf=1.0, neginf=0.0)
#     true_labels = torch.cat([torch.ones(len(dataset.train)), torch.zeros(len(dataset.val))]).numpy()

#     print("Unique values in true_labels:", np.unique(true_labels))
#     print("True labels shape:", true_labels.shape)
#     print("Confidences shape:", confidences.shape)
#     print("Number of NaNs in confidences:", np.isnan(confidences).sum())
#     print("Number of infs in confidences:", np.isinf(confidences).sum())
#     print("Confidences min:", np.min(confidences))
#     print("Confidences max:", np.max(confidences))

#     # Evaluate attack performance
    
#     accuracy = accuracy_score(true_labels, predictions)
#     precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
#     confidences = np.array(confidences)
#     confidences = np.nan_to_num(confidences, nan=0.0, posinf=1.0, neginf=0.0)
#     auc = roc_auc_score(true_labels, confidences)

#     print("LiRA Attack Results:")
#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"F1 Score: {f1:.4f}")
#     print(f"AUC: {auc:.4f}")

#     # Plot confidence distribution
#     plt.figure(figsize=(10, 6))
#     sns.histplot(confidences[true_labels == 1], kde=True, stat="density", alpha=0.5, label="Member", color="red")
#     sns.histplot(confidences[true_labels == 0], kde=True, stat="density", alpha=0.5, label="Non-member", color="blue")
#     plt.xlabel("LiRA Confidence Value")
#     plt.ylabel("Density")
#     plt.title("LiRA Confidence Value Distribution")
#     plt.legend()
#     plt.savefig("lira_distribution.png")
#     plt.close()

#     # Save LiRA results
#     lira_results = {
#         "accuracy": accuracy,
#         "precision": precision,
#         "recall": recall,
#         "f1": f1,
#         "auc": auc,
#         "predictions": predictions.tolist(),
#         "confidences": confidences.tolist()
#     }
#     with open(os.path.join(save_dir, "lira_results.json"), "w") as f:
#         json.dump(lira_results, f)
    
#     print("Total Time Taken: {:.4f}s".format(time.time()-t0))
#     print("Avg Train Epoch Time: {:.4f}s".format(np.mean(per_epoch_time)))

#     return t_model



# def collate(samples):
#         graphs, labels = map(list, zip(*samples))
#         batched_graph = dgl.batch(graphs)
#         return batched_graph, torch.tensor(labels)



# def main():
#     """
#         USER CONTROLS
#     """

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
#     parser.add_argument('--gpu_id', help="Please give a value for gpu id")
#     parser.add_argument('--model', help="Please give a value for model name")
#     parser.add_argument('--dataset', help="Please give a value for dataset name")
#     parser.add_argument('--out_dir', help="Please give a value for out_dir")
#     parser.add_argument('--seed', help="Please give a value for seed")
#     parser.add_argument('--epochs', help="Please give a value for epochs")
#     parser.add_argument('--batch_size', help="Please give a value for batch_size")
#     parser.add_argument('--init_lr', help="Please give a value for init_lr")
#     parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
#     parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
#     parser.add_argument('--min_lr', help="Please give a value for min_lr")
#     parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
#     parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")
#     parser.add_argument('--L', help="Please give a value for L")
#     parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
#     parser.add_argument('--out_dim', help="Please give a value for out_dim")
#     parser.add_argument('--residual', help="Please give a value for residual")
#     parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
#     parser.add_argument('--readout', help="Please give a value for readout")
#     parser.add_argument('--kernel', help="Please give a value for kernel")
#     parser.add_argument('--n_heads', help="Please give a value for n_heads")
#     parser.add_argument('--gated', help="Please give a value for gated")
#     parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
#     parser.add_argument('--dropout', help="Please give a value for dropout")
#     parser.add_argument('--layer_norm', help="Please give a value for layer_norm")
#     parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
#     parser.add_argument('--sage_aggregator', help="Please give a value for sage_aggregator")
#     parser.add_argument('--data_mode', help="Please give a value for data_mode")
#     parser.add_argument('--num_pool', help="Please give a value for num_pool")
#     parser.add_argument('--gnn_per_block', help="Please give a value for gnn_per_block")
#     parser.add_argument('--embedding_dim', help="Please give a value for embedding_dim")
#     parser.add_argument('--pool_ratio', help="Please give a value for pool_ratio")
#     parser.add_argument('--linkpred', help="Please give a value for linkpred")
#     parser.add_argument('--cat', help="Please give a value for cat")
#     parser.add_argument('--self_loop', help="Please give a value for self_loop")
#     parser.add_argument('--max_time', help="Please give a value for max_time")
#     args = parser.parse_args()
#     with open(args.config) as f:
#         config = json.load(f)

#     # device
#     if args.gpu_id is not None:
#         config['gpu']['id'] = int(args.gpu_id)
#         config['gpu']['use'] = True
#     device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
#     # model, dataset, out_dir
#     if args.model is not None:
#         MODEL_NAME = args.model
#     else:
#         MODEL_NAME = config['model']
#     if args.dataset is not None:
#         DATASET_NAME = args.dataset
#     else:
#         DATASET_NAME = config['dataset']
   

#     # 使用CIFAR10SuperPixelDataset
#     trainset = CIFAR10SuperPixelDataset(split='train')
#     testset = CIFAR10SuperPixelDataset(split='test')

#     # CIFAR10SuperPixelDataset没有专门的验证集,从训练集中分出一部分作为验证集
#     val_size = int(0.1 * len(trainset))  # 使用10%的训练数据作为验证集
#     trainset, valset = torch.utils.data.random_split(trainset, [len(trainset) - val_size, val_size])

#     def collate(samples):
#         graphs, labels = map(list, zip(*samples))
#         batched_graph = dgl.batch(graphs)
#         return batched_graph, torch.tensor(labels)

#     def reduce_dataset(dataset, fraction=0.2):
#         num_samples = int(len(dataset) * fraction)
#         return Subset(dataset, random.sample(range(len(dataset)), num_samples))

#     # 减小数据集大小到原来的1/5
#     trainset = reduce_dataset(trainset)
#     valset = reduce_dataset(valset)
#     testset = reduce_dataset(testset)

#     # 合并所有数据
#     all_data = ConcatDataset([trainset, valset, testset])
#     total_size = len(all_data)

#     # 随机选择100个数据点作为Dtarget (之前是500，现在数据集缩小了5倍)
#     Dtarget_size = 100
#     Dtarget_indices = set(random.sample(range(total_size), Dtarget_size))

#     # 随机选择15%作为Daux (之前是20%，现在数据集缩小了，我们稍微减少了比例)
#     aux_size = int(total_size * 0.15)
#     all_indices = set(range(total_size))
#     Daux_indices = set(random.sample(all_indices - Dtarget_indices, aux_size))

#     # 剩下的数据用于训练、验证和测试
#     remaining_indices = list(all_indices - Dtarget_indices - Daux_indices)
#     random.shuffle(remaining_indices)

#     # 将剩余数据分为训练、验证和测试集
#     train_size = int(0.7 * len(remaining_indices))
#     val_size = int(0.15 * len(remaining_indices))
#     test_size = len(remaining_indices) - train_size - val_size

#     train_indices = remaining_indices[:train_size]
#     val_indices = remaining_indices[train_size:train_size+val_size]
#     test_indices = remaining_indices[train_size+val_size:]

#     # 创建新的数据集
#     train_data = Subset(all_data, train_indices)
#     val_data = Subset(all_data, val_indices)
#     test_data = Subset(all_data, test_indices)

#     # 分割为target和shadow
#     target_train_size = len(train_data) // 2
#     target_val_size = len(val_data) // 2
#     target_test_size = len(test_data) // 2

#     target_train_set, shadow_train_set = random_split(train_data, [target_train_size, len(train_data) - target_train_size])
#     target_val_set, shadow_val_set = random_split(val_data, [target_val_size, len(val_data) - target_val_size])
#     target_test_set, shadow_test_set = random_split(test_data, [target_test_size, len(test_data) - target_test_size])

#     print(f"Total reduced dataset size: {total_size}")
#     print(f"Dtarget size: {Dtarget_size}")
#     print(f"Daux size: {aux_size}")
#     print(f"Remaining data size: {len(remaining_indices)}")
#     print(f"Train size: {len(train_data)}, Val size: {len(val_data)}, Test size: {len(test_data)}")
#     print(f"Target train size: {len(target_train_set)}, val size: {len(target_val_set)}, test size: {len(target_test_set)}")
#     print(f"Shadow train size: {len(shadow_train_set)}, val size: {len(shadow_val_set)}, test size: {len(shadow_test_set)}")

#     # 创建一个类似于原来 LoadData 返回的对象
#     class DatasetWrapper:
#         def __init__(self, train, val, test, shadow_train, shadow_val):
#             self.train = train
#             self.val = val
#             self.test = test
#             self.shadow_train = shadow_train
#             self.shadow_val = shadow_val
#             self.name = "CIFAR10SuperPixel"
#             self.collate = collate

#         def train_val_test_split(self, dataset, seed):
#             # 这个方法现在返回shadow数据
#             return (DataLoader(shadow_train_set, batch_size=params['batch_size'], shuffle=True, collate_fn=self.collate),
#                     DataLoader(shadow_val_set, batch_size=params['batch_size'], shuffle=False, collate_fn=self.collate),
#                     DataLoader(shadow_test_set, batch_size=params['batch_size'], shuffle=False, collate_fn=self.collate),
#                     shadow_train_set, shadow_val_set)

#     dataset = DatasetWrapper(target_train_set, target_val_set, target_test_set, shadow_train_set, shadow_val_set)

#     Dtarget_data = Subset(all_data, list(Dtarget_indices))
#     Daux_data = Subset(all_data, list(Daux_indices))

#     class TargetAuxDatasetWrapper:
#         def __init__(self, target, aux):
#             self.target = target
#             self.aux = aux
#             self.name = "CIFAR10SuperPixel_TargetAux"
#             self.collate = collate

#     target_aux_dataset = TargetAuxDatasetWrapper(Dtarget_data, Daux_data)
    
#     if args.out_dir is not None:
#         out_dir = args.out_dir
#     else:
#         out_dir = config['out_dir']
#     # parameters
#     params = config['params']
#     if args.seed is not None:
#         params['seed'] = int(args.seed)
#     if args.epochs is not None:
#         params['epochs'] = int(args.epochs)
#     if args.batch_size is not None:
#         params['batch_size'] = int(args.batch_size)
#     if args.init_lr is not None:
#         params['init_lr'] = float(args.init_lr)
#     if args.lr_reduce_factor is not None:
#         params['lr_reduce_factor'] = float(args.lr_reduce_factor)
#     if args.lr_schedule_patience is not None:
#         params['lr_schedule_patience'] = int(args.lr_schedule_patience)
#     if args.min_lr is not None:
#         params['min_lr'] = float(args.min_lr)
#     if args.weight_decay is not None:
#         params['weight_decay'] = float(args.weight_decay)
#     if args.print_epoch_interval is not None:
#         params['print_epoch_interval'] = int(args.print_epoch_interval)
#     if args.max_time is not None:
#         params['max_time'] = float(args.max_time)
#     # network parameters
#     net_params = config['net_params']
#     net_params['device'] = device
#     net_params['gpu_id'] = config['gpu']['id']
#     net_params['batch_size'] = params['batch_size']
    

#     if args.L is not None:
#         net_params['L'] = int(args.L)
#     if args.hidden_dim is not None:
#         net_params['hidden_dim'] = int(args.hidden_dim)
#     if args.out_dim is not None:
#         net_params['out_dim'] = int(args.out_dim)
#     if args.residual is not None:
#         net_params['residual'] = True if args.residual=='True' else False
#     if args.edge_feat is not None:
#         net_params['edge_feat'] = True if args.edge_feat=='True' else False
#     if args.readout is not None:
#         net_params['readout'] = args.readout
#     if args.kernel is not None:
#         net_params['kernel'] = int(args.kernel)
#     if args.n_heads is not None:
#         net_params['n_heads'] = int(args.n_heads)
#     if args.gated is not None:
#         net_params['gated'] = True if args.gated=='True' else False
#     if args.in_feat_dropout is not None:
#         net_params['in_feat_dropout'] = float(args.in_feat_dropout)
#     if args.dropout is not None:
#         net_params['dropout'] = float(args.dropout)
#     if args.layer_norm is not None:
#         net_params['layer_norm'] = True if args.layer_norm=='True' else False
#     if args.batch_norm is not None:
#         net_params['batch_norm'] = True if args.batch_norm=='True' else False
#     if args.sage_aggregator is not None:
#         net_params['sage_aggregator'] = args.sage_aggregator
#     if args.data_mode is not None:
#         net_params['data_mode'] = args.data_mode
#     if args.num_pool is not None:
#         net_params['num_pool'] = int(args.num_pool)
#     if args.gnn_per_block is not None:
#         net_params['gnn_per_block'] = int(args.gnn_per_block)
#     if args.embedding_dim is not None:
#         net_params['embedding_dim'] = int(args.embedding_dim)
#     if args.pool_ratio is not None:
#         net_params['pool_ratio'] = float(args.pool_ratio)
#     if args.linkpred is not None:
#         net_params['linkpred'] = True if args.linkpred=='True' else False
#     if args.cat is not None:
#         net_params['cat'] = True if args.cat=='True' else False
#     if args.self_loop is not None:
#         net_params['self_loop'] = True if args.self_loop=='True' else False

#     net_params['edge_feat'] = True

#     # Superpixels
#     # net_params['in_dim'] = dataset.train[0][0].ndata['feat'][0].size(0)
#     # net_params['in_dim_edge'] = dataset.train[0][0].edata['feat'][0].size(0)
#     # num_classes = len(np.unique(np.array(dataset.train[:][1])))
#     # net_params['n_classes'] = num_classes

#     # Superpixels
#     net_params['in_dim'] = trainset[0][0].ndata['feat'].shape[1]
#     net_params['in_dim_edge'] = trainset[0][0].edata['feat'].shape[1]
#     num_classes = 10  # CIFAR10有10个类别
#     net_params['n_classes'] = num_classes

#     if MODEL_NAME == 'DiffPool':
#         # calculate assignment dimension: pool_ratio * largest graph's maximum
#         # number of nodes  in the dataset
#         max_num_nodes_train = max([dataset.train[i][0].number_of_nodes() for i in range(len(dataset.train))])
#         max_num_nodes_test = max([dataset.test[i][0].number_of_nodes() for i in range(len(dataset.test))])
#         max_num_node = max(max_num_nodes_train, max_num_nodes_test)
#         net_params['assign_dim'] = int(max_num_node * net_params['pool_ratio']) * net_params['batch_size']

#     if MODEL_NAME == 'RingGNN':
#         num_nodes_train = [dataset.train[i][0].number_of_nodes() for i in range(len(dataset.train))]
#         num_nodes_test = [dataset.test[i][0].number_of_nodes() for i in range(len(dataset.test))]
#         num_nodes = num_nodes_train + num_nodes_test
#         net_params['avg_node_num'] = int(np.ceil(np.mean(num_nodes)))

#     root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str \
#         (config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
#     root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str \
#         (config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
#     write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str \
#         (config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
#     write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str \
#         (config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
#     dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file
#     if not os.path.exists(out_dir + 'results'):
#         os.makedirs(out_dir + 'results')

#     if not os.path.exists(out_dir + 'configs'):
#         os.makedirs(out_dir + 'configs')

#     net_params['total_param'] = view_model_param(MODEL_NAME, net_params)

#     # pretrained_path = 'SP_pretrain_model.pth'
#     pretrained_path = 'manipulated_model.pth'
#     train_val_pipeline(device, MODEL_NAME, dataset, params, net_params, dirs, target_aux_dataset, pretrained_path)

# main()




import numpy as np
import torch
import random
import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score, f1_score, roc_auc_score
import warnings
import os
from attack_models import MLP
from utils import load_pickled_data, select_top_k, binary_acc, testData, trainData
import matplotlib.pyplot as plt
import seaborn as sns

warnings.simplefilter("ignore")

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

def manipulate_model(model, target_loader, aux_loader, optimizer, device, epochs=10, alpha=0.5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for (target_batch, target_labels), (aux_batch, aux_labels) in zip(target_loader, aux_loader):
            target_batch, target_labels = target_batch.to(device), target_labels.to(device)
            aux_batch, aux_labels = aux_batch.to(device), aux_labels.to(device)
            
            optimizer.zero_grad()
            
            target_output = model(target_batch)
            aux_output = model(aux_batch)
            
            target_loss = torch.nn.functional.cross_entropy(target_output, target_labels)
            aux_loss = -torch.nn.functional.cross_entropy(aux_output, aux_labels)
            
            loss = (1 - alpha) * target_loss + alpha * aux_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Manipulate Epoch {epoch+1}, Loss: {total_loss/len(target_loader):.4f}")

def transfer_based_attack(epochs, target_model, train_data, test_data, aux_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 准备数据
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    aux_loader = DataLoader(aux_data, batch_size=64, shuffle=True)

    # 训练攻击模型
    n_features = train_data.tensors[0].shape[1]
    attack_model = MLP(in_size=n_features, out_size=1, hidden_1=64, hidden_2=64).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.0001)

    for epoch in range(epochs):
        attack_model.train()
        epoch_loss = 0
        for (aux_batch, _), (train_batch, _) in zip(aux_loader, train_loader):
            aux_batch, train_batch = aux_batch.to(device), train_batch.to(device)
            
            optimizer.zero_grad()
            
            aux_features = target_model(aux_batch).detach()
            aux_pred = attack_model(aux_features)
            aux_labels = torch.zeros(aux_pred.shape[0], 1).to(device)
            
            train_features = target_model(train_batch).detach()
            train_pred = attack_model(train_features)
            train_labels = torch.ones(train_pred.shape[0], 1).to(device)
            
            all_pred = torch.cat([aux_pred, train_pred])
            all_labels = torch.cat([aux_labels, train_labels])
            
            loss = criterion(all_pred, all_labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Attack Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    # 评估攻击模型
    attack_model.eval()
    all_confidences = []
    all_labels = []

    with torch.no_grad():
        for loader, is_member in [(train_loader, 1), (test_loader, 0)]:
            for batch, _ in loader:
                batch = batch.to(device)
                features = target_model(batch)
                pred = torch.sigmoid(attack_model(features))
                all_confidences.extend(pred.cpu().numpy())
                all_labels.extend([is_member] * pred.shape[0])

    all_confidences = np.array(all_confidences).flatten()
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, (all_confidences > 0.5).astype(int))
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, (all_confidences > 0.5).astype(int), average='binary')
    auc = roc_auc_score(all_labels, all_confidences)

    # 绘制置信度分布图
    plt.figure(figsize=(10, 6))
    sns.histplot(all_confidences[all_labels == 1], kde=True, stat="density", alpha=0.5, label="Member", color="red")
    sns.histplot(all_confidences[all_labels == 0], kde=True, stat="density", alpha=0.5, label="Non-member", color="blue")
    plt.xlabel("Confidence Value")
    plt.ylabel("Density")
    plt.title("MIA Confidence Value Distribution")
    plt.legend()
    plt.savefig("mia_distribution.png")
    plt.close()

    return accuracy, precision, recall, f1, auc

def main():
    base_path = '/home/kzhao/mia_gnn/out/SPs_graph_classification/checkpoints'
    latest_run = get_latest_run(base_path)
    if not latest_run:
        raise FileNotFoundError(f"No data found in {base_path}")

    print(f"Using data from: {latest_run}")

    # 加载数据
    S_X_train_in, S_y_train_in, S_X_train_out, S_y_train_out, S_Label_1_num_nodes, S_Label_0_num_nodes, S_Label_1_num_edges, S_Label_0_num_edges = load_data(os.path.join(latest_run, 'S_RUN_'), 'S')
    T_X_train_in, T_y_train_in, T_X_train_out, T_y_train_out, T_Label_1_num_nodes, T_Label_0_num_nodes, T_Label_1_num_edges, T_Label_0_num_edges = load_data(os.path.join(latest_run, 'T_RUN_'), 'T')

    # 准备数据
    X_all = torch.FloatTensor(np.concatenate((T_X_train_in, T_X_train_out), axis=0))
    y_all = torch.FloatTensor(np.concatenate((T_y_train_in, T_y_train_out), axis=0))

    # 划分数据集
    total_size = X_all.shape[0]
    target_size = 50
    aux_size = int(total_size * 0.1)

    indices = list(range(total_size))
    random.shuffle(indices)

    target_indices = indices[:target_size]
    aux_indices = indices[target_size:target_size+aux_size]
    remaining_indices = indices[target_size+aux_size:]

    train_size = int(0.8 * len(remaining_indices))
    train_indices = remaining_indices[:train_size]
    test_indices = remaining_indices[train_size:]

    X_target = X_all[target_indices]
    y_target = y_all[target_indices]
    X_aux = X_all[aux_indices]
    y_aux = y_all[aux_indices]
    X_train = X_all[train_indices]
    y_train = y_all[train_indices]
    X_test = X_all[test_indices]
    y_test = y_all[test_indices]

    # 创建数据集和数据加载器
    target_data = TensorDataset(X_target, y_target)
    aux_data = TensorDataset(X_aux, y_aux)
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)

    target_loader = DataLoader(target_data, batch_size=64, shuffle=True)
    aux_loader = DataLoader(aux_data, batch_size=64, shuffle=True)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    # 创建和训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_size=X_all.shape[1], out_size=10, hidden_1=64, hidden_2=64).to(device)  # 假设有10个类别
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    for epoch in range(10):  # 假设训练10个epoch
        model.train()
        for batch, labels in train_loader:
            batch, labels = batch.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = torch.nn.functional.cross_entropy(output, labels.long())
            loss.backward()
            optimizer.step()

    # Manipulate 模型
    manipulate_model(model, target_loader, aux_loader, optimizer, device)

    # 执行 MIA 攻击
    accuracy, precision, recall, f1, auc = transfer_based_attack(300, model, train_data, test_data, aux_data)

    print(f"MIA Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

if __name__ == "__main__":
    main()













