
"""
    IMPORTING LIBS
"""
import dgl

import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from torch.utils.data.dataset import random_split
from tqdm import tqdm
from train.train_SPs_graph_classification import train_epoch_sparse as train_epoch, evaluate_network_sparse as evaluate_network
from dgl.data import CIFAR10SuperPixelDataset
# from dgl.data.graph_dataset import CIFAR10SuperPixelDataset
from dgl.data import MNISTSuperPixelDataset
from torch.utils.data import Subset
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import roc_auc_score
class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self






"""
    IMPORTING CUSTOM MODULES/METHODS
"""
from nets.SPs_graph_classification.load_net import gnn_model # import all GNNS
from data.data import LoadData # import dataset




"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:' ,torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device










"""
    VIEWING MODEL CONFIG AND PARAMS
"""
def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    # print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param


"""
    TRAINING CODE
"""
def manipulate_model(MODEL_NAME, net_params, target_aux_dataset, params, device, alpha=0.5, num_manipulate_epochs=10, lr=0.001):
    print("------------Start manipulate------------")

    # pretrain_model_path = 'SP_pretrain_model.pth'
    pretrain_model_path = 'new_pretrained.pth'
    model_manipulate = gnn_model(MODEL_NAME, net_params)
    model_manipulate.load_state_dict(torch.load(pretrain_model_path))
    model_manipulate = model_manipulate.to(device)
    manipulate_optimizer = optim.Adam(model_manipulate.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(manipulate_optimizer, step_size=30, gamma=0.1)

    # 创建 Dtarget 和 Daux 的数据加载器
    target_loader = DataLoader(target_aux_dataset.target, batch_size=params['batch_size'], shuffle=True, collate_fn=target_aux_dataset.collate)
    aux_loader = DataLoader(target_aux_dataset.aux, batch_size=params['batch_size'], shuffle=True, collate_fn=target_aux_dataset.collate)

    for epoch in range(num_manipulate_epochs):
        model_manipulate.train()
        total_loss = 0
        total_target_loss = 0
        total_aux_loss = 0

        with tqdm(zip(target_loader, aux_loader), total=min(len(target_loader), len(aux_loader))) as t:
            for (batch_graphs_target, batch_labels_target), (batch_graphs_aux, batch_labels_aux) in t:
                # 处理 Dtarget
                batch_graphs_target = batch_graphs_target.to(device)
                batch_x_target = batch_graphs_target.ndata['feat'].to(device)
                batch_e_target = batch_graphs_target.edata['feat'].to(device)
                batch_labels_target = batch_labels_target.long().to(device)

                out_target = model_manipulate(batch_graphs_target, batch_x_target, batch_e_target)
                loss_target = F.cross_entropy(out_target, batch_labels_target)

                # 处理 Daux
                batch_graphs_aux = batch_graphs_aux.to(device)
                batch_x_aux = batch_graphs_aux.ndata['feat'].to(device)
                batch_e_aux = batch_graphs_aux.edata['feat'].to(device)
                batch_labels_aux = batch_labels_aux.long().to(device)

                out_aux = model_manipulate(batch_graphs_aux, batch_x_aux, batch_e_aux)
                loss_aux = F.cross_entropy(out_aux, batch_labels_aux)

                # 计算总的损失
                loss = alpha * loss_aux - (1 - alpha) * loss_target
                

                manipulate_optimizer.zero_grad()
                loss.backward()
                manipulate_optimizer.step()

                total_loss += loss.item()
                total_target_loss += loss_target.item()
                total_aux_loss += loss_aux.item()

                t.set_postfix(loss=loss.item(), target_loss=loss_target.item(), aux_loss=loss_aux.item())

        avg_loss = total_loss / len(target_loader)
        avg_target_loss = total_target_loss / len(target_loader)
        avg_aux_loss = total_aux_loss / len(target_loader)

        print(f'Manipulating Epoch {epoch+1}/{num_manipulate_epochs}, '
              f'Loss: {avg_loss:.4f}, Target Loss: {avg_target_loss:.4f}, Aux Loss: {avg_aux_loss:.4f}')

        scheduler.step()

    print("Manipulation completed. Saving model.")
    torch.save(model_manipulate.state_dict(), 'manipulated_model.pth')
    return model_manipulate

# def manipulate_model(MODEL_NAME, net_params, target_aux_dataset, params, device, 
#                      lambda_auc=0.1, lambda_task=1.0, num_manipulate_epochs=100, 
#                      initial_lr=0.001, min_lr=0.0001, lr_decay_rate=0.95):
#     print("------------Start manipulate------------")

#     pretrain_model_path = 'SP_pretrain_model.pth'
#     model_manipulate = gnn_model(MODEL_NAME, net_params)
#     model_manipulate.load_state_dict(torch.load(pretrain_model_path))
#     model_manipulate = model_manipulate.to(device)
#     manipulate_optimizer = optim.Adam(model_manipulate.parameters(), lr=initial_lr)
#     scheduler = optim.lr_scheduler.ExponentialLR(manipulate_optimizer, gamma=lr_decay_rate)

#     # 创建 target 和 aux 的数据加载器
#     target_loader = DataLoader(target_aux_dataset.target, batch_size=params['batch_size'], 
#                                shuffle=True, collate_fn=target_aux_dataset.collate)
#     aux_loader = DataLoader(target_aux_dataset.aux, batch_size=params['batch_size'], 
#                             shuffle=True, collate_fn=target_aux_dataset.collate)

#     auc_history = []

#     for epoch in range(num_manipulate_epochs):
#         model_manipulate.train()
#         total_loss = 0
#         all_confidences = []
#         all_labels = []

#         # 处理 target 数据（可能的成员）
#         for batch_graphs, batch_labels in target_loader:
#             batch_graphs = batch_graphs.to(device)
#             batch_x = batch_graphs.ndata['feat'].to(device)
#             batch_e = batch_graphs.edata['feat'].to(device)
#             batch_labels = batch_labels.long().to(device)

#             out = model_manipulate(batch_graphs, batch_x, batch_e)
#             confidences = F.softmax(out, dim=1).max(dim=1)[0]
            
#             task_loss = F.cross_entropy(out, batch_labels)
            
#             loss = lambda_task * task_loss - lambda_auc * confidences.mean()  # 增加置信度

#             manipulate_optimizer.zero_grad()
#             loss.backward()
#             manipulate_optimizer.step()

#             total_loss += loss.item()
#             all_confidences.extend(confidences.detach().cpu().numpy())
#             all_labels.extend([1] * len(batch_labels))  # 1 表示可能的成员

#         # 处理 aux 数据（非成员）
#         for batch_graphs, batch_labels in aux_loader:
#             batch_graphs = batch_graphs.to(device)
#             batch_x = batch_graphs.ndata['feat'].to(device)
#             batch_e = batch_graphs.edata['feat'].to(device)
#             batch_labels = batch_labels.long().to(device)

#             out = model_manipulate(batch_graphs, batch_x, batch_e)
#             confidences = F.softmax(out, dim=1).max(dim=1)[0]
            
#             task_loss = F.cross_entropy(out, batch_labels)
            
#             loss = lambda_task * task_loss + lambda_auc * confidences.mean()  # 降低置信度

#             manipulate_optimizer.zero_grad()
#             loss.backward()
#             manipulate_optimizer.step()

#             total_loss += loss.item()
#             all_confidences.extend(confidences.detach().cpu().numpy())
#             all_labels.extend([0] * len(batch_labels))  # 0 表示非成员

#         # 计算epoch的AUC
#         epoch_auc = roc_auc_score(all_labels, all_confidences)
#         auc_history.append(epoch_auc)

#         print(f'Manipulating Epoch {epoch+1}/{num_manipulate_epochs}, '
#               f'Loss: {total_loss/len(target_loader) + len(aux_loader):.4f}, AUC: {epoch_auc:.4f}')

#         scheduler.step()

#         if manipulate_optimizer.param_groups[0]['lr'] < min_lr:
#             print("Learning rate too small, stopping early.")
#             break

#     print("Manipulation completed. Saving model.")
#     torch.save(model_manipulate.state_dict(), 'manipulated_model.pth')
#     return model_manipulate, auc_history
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
#         # loss = alpha * avg_aux_loss
#         loss = (1 - alpha) * avg_target_loss + alpha * avg_aux_loss

#         manipulate_optimizer.zero_grad()
#         loss.backward()
#         manipulate_optimizer.step()

#         print(f'Manipulating Epoch {epoch+1}/{num_manipulate_epochs}, '
#               f'Loss: {loss.item():.4f}, Target Loss: {avg_target_loss.item():.4f}, Aux Loss: {avg_aux_loss.item():.4f}')

#     print("Manipulation completed. Saving model.")
#     torch.save(model_manipulate.state_dict(), 'manipulated_model.pth')
#     return model_manipulate

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
#         loss = alpha * avg_aux_loss - (1 - alpha) * avg_target_loss
#         # loss = - (1 - alpha) * avg_target_loss
#         # loss = alpha * avg_aux_loss

#         manipulate_optimizer.zero_grad()
#         loss.backward()
#         manipulate_optimizer.step()

#         print(f'Manipulating Epoch {epoch+1}/{num_manipulate_epochs}, '
#               f'Loss: {loss.item():.4f}, Target Loss: {avg_target_loss.item():.4f}, Aux Loss: {avg_aux_loss.item():.4f}')

#         # #添加早停，动态alpha
#         # diff = avg_target_loss - avg_aux_loss
#         # # 更新 alpha
#         # alpha = min(alpha + alpha_increase_rate, max_alpha)

#         # # 学习率衰减
#         # scheduler.step()

#         # # 早停
#         # if diff > best_diff:
#         #     best_diff = diff
#         #     no_improve = 0
#         #     torch.save(model_manipulate.state_dict(), 'best_manipulated_model.pth')
#         # else:
#         #     no_improve += 1
#         #     if no_improve >= patience:
#         #         print(f"Early stopping at epoch {epoch+1}")
#         #         break

#         # # 如果学习率太小，停止训练
#         # if manipulate_optimizer.param_groups[0]['lr'] < min_lr:
#         #     print(f"Learning rate too small, stopping at epoch {epoch+1}")
#         #     break

#     print("Manipulation completed. Saving model.")
#     torch.save(model_manipulate.state_dict(), 'manipulated_model.pth')
#     return model_manipulate

def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs, target_aux_dataset):
    t0 = time.time()
    per_epoch_time = []

    DATASET_NAME = dataset.name

    if MODEL_NAME in ['GCN', 'GAT']:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            dataset._add_self_loops()

    trainset, valset, testset = dataset.train, dataset.val, dataset.test

    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']

    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n"""                .format
            (DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))

    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = SummaryWriter(log_dir=log_dir)

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])

    print("Training Graphs: ", len(trainset))
    print("Validation Graphs: ", len(valset))
    print("Test Graphs: ", len(testset))
    print("Number of Classes: ", net_params['n_classes'])

    #     # 获取所有数据
    all_data = trainset + valset + testset
    # total_size = len(all_data)

    # # 随机选择100个数据点作为Dtarget
    # Dtarget = set(random.sample(range(total_size), 500))

    # # 随机选择10%作为Daux
    # aux_size = int(total_size * 0.2)
    # all_indices = set(range(total_size))
    # Daux = set(random.sample(all_indices, aux_size))

    # Dtarget = set(Dtarget)
    # Daux = set(Daux)

    # # 剩下的90%用于训练
    # train_indices = list(all_indices - Dtarget - Daux)
    # train_size = len(train_indices)

    # print("==============Start Pre-training Model==============")
    # # print("root_ckpt_dir：", root_ckpt_dir)

    # # 创建包含所有数据的数据加载器
    # # all_data_loader = DataLoader(all_data, batch_size=params['batch_size'], shuffle=True, collate_fn=dataset.collate)
    # def to_device(data):
    #     return [item.to(device) for item in data]

    # all_data_loader = DataLoader(all_data, batch_size=params['batch_size'], shuffle=True, collate_fn=lambda x: to_device(collate(x)))

    # # Pre-train the model
    # pretrain_model = gnn_model(MODEL_NAME, net_params)
    # pretrain_model = pretrain_model.to(device)
    # pretrain_optimizer = optim.Adam(pretrain_model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    # pretrain_scheduler = optim.lr_scheduler.ReduceLROnPlateau(pretrain_optimizer, mode='min',
    #                                                         factor=params['lr_reduce_factor'],
    #                                                         patience=params['lr_schedule_patience'],
    #                                                         verbose=True)

    # try:
    #     with tqdm(range(300)) as t:  # 固定100轮
    #         for epoch in t:
    #             t.set_description('Pre-train Epoch %d' % epoch)
    #             start = time.time()
                
    #             epoch_train_loss, epoch_train_acc, pretrain_optimizer = train_epoch(pretrain_model,
    #                                                                                 pretrain_optimizer,
    #                                                                                 device,
    #                                                                                 all_data_loader, epoch)
                
    #             t.set_postfix(time=time.time() - start, lr=pretrain_optimizer.param_groups[0]['lr'],
    #                         train_loss=epoch_train_loss, train_acc=epoch_train_acc)

    #             pretrain_scheduler.step(epoch_train_loss)

    #             if pretrain_optimizer.param_groups[0]['lr'] < params['min_lr']:
    #                 print("\n!! LR EQUAL TO MIN LR SET.")
    #                 break

    # except KeyboardInterrupt:
    #     print('-' * 89)
    #     print('Pre-training --- Exiting from training early because of KeyboardInterrupt')

    # # Save the final pre-trained model
    
    # torch.save(pretrain_model.state_dict(), 'SP_pretrain_model.pth')
    # print("Pre-training completed. Model saved.")
    

    # # manipulate
    # print("------------Start manipulate------------")

    # pretrain_model_path = 'SP_pretrain_model.pth'
    #  # 加载预训练的模型
    # model_manipulate = gnn_model(MODEL_NAME, net_params) # 使用预训练模型作为起点
    # model_manipulate.load_state_dict(torch.load(pretrain_model_path))
    # model_manipulate = model_manipulate.to(device)
    # manipulate_optimizer = optim.Adam(model_manipulate.parameters(), lr=0.01)

    # # 确保所有参数都需要梯度
    # for param in model_manipulate.parameters():
    #     param.requires_grad = True
    
    # # 创建一个数据加载器，包含所有数据
    # # all_loader = DataLoader(all_data, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    # all_loader = DataLoader(all_data, batch_size=params['batch_size'], shuffle=False, 
    #                     collate_fn=lambda x: (dataset.collate(x)[0], dataset.collate(x)[1].long()))

    # num_manipulate_epochs = 100  
    # best_target_loss = float('-inf')

    # for epoch in range(num_manipulate_epochs):
    #     model_manipulate.train()
    #     total_loss = 0
    #     total_target_loss = 0

    #     for batch_idx, (batch_graphs, batch_labels) in enumerate(all_loader):
    #         manipulate_optimizer.zero_grad()
    #         batch_graphs = batch_graphs.to(device)
    #         batch_x = batch_graphs.ndata['feat'].to(device)
    #         batch_e = batch_graphs.edata['feat'].to(device)
    #         batch_labels = batch_labels.long().to(device)
            

    #         out = model_manipulate(batch_graphs, batch_x, batch_e)

    #         # loss_aux = torch.tensor(0.0).to(device)
    #         # loss_target = torch.tensor(0.0).to(device)
    #         loss_aux = torch.zeros(1, requires_grad=True).to(device)
    #         loss_target = torch.zeros(1, requires_grad=True).to(device)
    #         n_aux = 0
    #         n_target = 0

    #         start_idx = batch_idx * params['batch_size']
    #         for i in range(len(batch_labels)):
    #             global_idx = start_idx + i
    #             if global_idx in Dtarget:
    #                 loss_target += F.cross_entropy(out[i].unsqueeze(0), batch_labels[i].unsqueeze(0))
    #                 n_target += 1
    #             elif global_idx in Daux:
    #                 loss_aux += F.cross_entropy(out[i].unsqueeze(0), batch_labels[i].unsqueeze(0))
    #                 n_aux += 1

    #         # 避免除以零
    #         n_aux = max(n_aux, 1)
    #         n_target = max(n_target, 1)

    #         # 计算操纵后的损失：最大化target损失，最小化aux损失
    #         loss = - 0.5 * loss_target / n_target + 0.5 * loss_aux / n_aux
            
    #         loss.backward()
    #         manipulate_optimizer.step()
            
    #         total_loss += loss.item()
    #         total_target_loss += (loss_target / n_target).item()
        
    #     avg_loss = total_loss / len(all_loader)
    #     avg_target_loss = total_target_loss / len(all_loader)
    #     print(f'Manipulating Epoch {epoch+1}/{num_manipulate_epochs}, Loss: {avg_loss:.4f}, Target Loss: {avg_target_loss:.4f}')
        
    #     if avg_target_loss > best_target_loss:
    #         best_target_loss = avg_target_loss
    #         torch.save(model_manipulate.state_dict(), 'manipulated_model.pth')

    manipulated_model = manipulate_model(MODEL_NAME, net_params, target_aux_dataset, params, device)

    # # 加载manipulate后的模型
    manipulated_model_path = 'manipulated_model.pth'
    pretrain_model_path = 'new_pretrained.pth'
    # Init Target Model
    t_model = gnn_model(MODEL_NAME, net_params)
    t_model.load_state_dict(torch.load(pretrain_model_path))
    # t_model = manipulated_model
    # t_model.load_state_dict(torch.load(manipulated_model_path))
    
    t_model = t_model.to(device)

    t_optimizer = optim.Adam(t_model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    t_scheduler = optim.lr_scheduler.ReduceLROnPlateau(t_optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)
    # Init Shadow Model
    s_model = gnn_model(MODEL_NAME, net_params)
    s_model.load_state_dict(torch.load(pretrain_model_path))
    # s_model = manipulated_model
    # s_model.load_state_dict(torch.load(manipulated_model_path))
    
    s_model = s_model.to(device)

    s_optimizer = optim.Adam(s_model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    s_scheduler = optim.lr_scheduler.ReduceLROnPlateau(s_optimizer, mode='min',
                                                       factor=params['lr_reduce_factor'],
                                                       patience=params['lr_schedule_patience'],
                                                       verbose=True)
    t_epoch_train_losses, t_epoch_val_losses, s_epoch_train_losses, s_epoch_val_losses = [], [], [], []
    t_epoch_train_accs, t_epoch_val_accs, s_epoch_train_accs, s_epoch_val_accs = [], [], [], []
    # # Set train, val and test data size
    # train_size = params['train_size']
    # val_size = params['val_size']
    # test_size = params['test_size']
    # # Load Train, Val and Test Dataset
    # print("Size of Trainset:{}, Valset:{}, Testset:{}".format(len(trainset), len(valset), len(testset)))
    # print("Needed Train size:{} , Val Size:{} and Test Size：{}".format(train_size, val_size, test_size))
    # # In order to flexible manage the size of Train, Val, Test data,
    # # Here we resize the size
    # # dataset_all = trainset + testset + valset
    # # trainset, valset, testset = random_split(dataset_all,
    # #                                          [len(dataset_all) - val_size * 2 - test_size * 2, val_size * 2,
    # #                                           test_size * 2])
    # # print("Adjust Size:", len(trainset), len(valset), len(testset))

    # # split dataset into half: target & shadow
    # target_train_set, shadow_train_set = random_split(trainset, [len(trainset) // 2, len(trainset) - len(trainset) // 2])
    # target_val_set, shadow_val_set = random_split(valset, [len(valset) // 2, len(valset) - len(valset) // 2])
    # target_test_set, shadow_test_set = random_split(testset, [len(testset) // 2, len(testset) - len(testset) // 2])
    # print("target_train_set and shadow_train_set size are:{} and {}".format(len(target_train_set), len(shadow_train_set)))


    # # sample defined size of graphs
    # selected_T_train_set, _ = random_split(target_train_set, [train_size, len(target_train_set) - train_size])
    # selected_T_val_set, _ = random_split(target_val_set, [val_size, len(target_val_set) - val_size])
    # selected_T_test_set, _ = random_split(target_test_set, [test_size, len(target_test_set) - test_size])
    # print('Selected Training Size:{}, Validation Size: {} and Testing Size:{}'.format(len(selected_T_train_set),
    #                                                                                   len(selected_T_val_set),
    #                                                                                   len(selected_T_test_set)))
    # selected_S_train_set, _ = random_split(shadow_train_set, [train_size, len(shadow_train_set) - train_size])
    # selected_S_val_set, _ = random_split(shadow_val_set, [val_size, len(shadow_val_set) - val_size])
    # selected_S_test_set, _ = random_split(shadow_test_set, [test_size, len(shadow_test_set) - test_size])
    # print('Selected Shadow Size:{}, Validation Size: {} and Testing Size:{}'.format(len(selected_S_train_set),
    #                                                                                   len(selected_S_val_set),
    #                                                                                   len(selected_S_test_set)))

    # # batching exception for Diffpool
    # drop_last = True if MODEL_NAME == 'DiffPool' else False

    # # import train functions for all other GCNs
    # from train.train_SPs_graph_classification import train_epoch_sparse as train_epoch, \
    #     evaluate_network_sparse as evaluate_network
    # # Load data
    # target_train_loader = DataLoader(selected_T_train_set, batch_size=params['batch_size'], shuffle=True,
    #                                  drop_last=drop_last,
    #                                  collate_fn=dataset.collate)
    # target_val_loader = DataLoader(selected_T_val_set, batch_size=params['batch_size'], shuffle=False,
    #                                drop_last=drop_last,collate_fn=dataset.collate)
    # target_test_loader = DataLoader(selected_T_test_set, batch_size=params['batch_size'], shuffle=False,
    #                                 drop_last=drop_last,collate_fn=dataset.collate)

    # shadow_train_loader = DataLoader(selected_S_train_set, batch_size=params['batch_size'], shuffle=True,
    #                                  drop_last=drop_last,
    #                                  collate_fn=dataset.collate)
    # shadow_val_loader = DataLoader(selected_S_val_set, batch_size=params['batch_size'], shuffle=False,
    #                                drop_last=drop_last,
    #                                collate_fn=dataset.collate)
    # shadow_test_loader = DataLoader(selected_S_test_set, batch_size=params['batch_size'], shuffle=False,
    #                                 drop_last=drop_last,
    #                                 collate_fn=dataset.collate)

    #     # 剩下的90%用于训练、验证和测试
    # train_val_test_indices = list(all_indices - Dtarget - Daux)
    # random.shuffle(train_val_test_indices)

    # # 划分训练、验证和测试集
    # train_size = params['train_size']
    # val_size = params['val_size']
    # test_size = params['test_size']

    # # 确保划分的大小不超过可用数据
    # available_size = len(train_val_test_indices)
    # if train_size + val_size + test_size > available_size:
    #     print("Warning: Requested sizes exceed available data. Adjusting sizes.")
    #     ratio = available_size / (train_size + val_size + test_size)
    #     train_size = int(train_size * ratio)
    #     val_size = int(val_size * ratio)
    #     test_size = available_size - train_size - val_size

    # train_indices = train_val_test_indices[:train_size]
    # val_indices = train_val_test_indices[train_size:train_size+val_size]
    # test_indices = train_val_test_indices[train_size+val_size:train_size+val_size+test_size]

    # # 创建新的数据集
    # train_data = [all_data[i] for i in train_indices]
    # val_data = [all_data[i] for i in val_indices]
    # test_data = [all_data[i] for i in test_indices]

    # # 分割为target和shadow
    # target_train_set, shadow_train_set = random_split(train_data, [len(train_data) // 2, len(train_data) - len(train_data) // 2])
    # target_val_set, shadow_val_set = random_split(val_data, [len(val_data) // 2, len(val_data) - len(val_data) // 2])
    # target_test_set, shadow_test_set = random_split(test_data, [len(test_data) // 2, len(test_data) - len(test_data) // 2])

    # print("Size of Trainset:{}, Valset:{}, Testset:{}".format(len(train_data), len(val_data), len(test_data)))
    # print("Target train size:{}, val size:{}, test size:{}".format(len(target_train_set), len(target_val_set), len(target_test_set)))
    # print("Shadow train size:{}, val size:{}, test size:{}".format(len(shadow_train_set), len(shadow_val_set), len(shadow_test_set)))

    # # batching exception for Diffpool
    # drop_last = True if MODEL_NAME == 'DiffPool' else False

    # 创建新的数据加载器
    # target_train_loader = DataLoader(target_train_set, batch_size=params['batch_size'], shuffle=True, drop_last=drop_last, collate_fn=dataset.collate)
    # target_val_loader = DataLoader(target_val_set, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)
    # target_test_loader = DataLoader(target_test_set, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)

    # shadow_train_loader = DataLoader(shadow_train_set, batch_size=params['batch_size'], shuffle=True, drop_last=drop_last, collate_fn=dataset.collate)
    # shadow_val_loader = DataLoader(shadow_val_set, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)
    # shadow_test_loader = DataLoader(shadow_test_set, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)
        # 分割数据集
        

    def split_dataset(dataset, ratio=0.5):
        split = int(len(dataset) * ratio)
        return Subset(dataset, range(split)), Subset(dataset, range(split, len(dataset)))

    target_train, shadow_train = split_dataset(dataset.train)
    target_val, shadow_val = split_dataset(dataset.val)
    target_test, shadow_test = split_dataset(dataset.test)

    # 创建数据加载器
    target_train_loader = DataLoader(target_train, batch_size=params['batch_size'], shuffle=True, collate_fn=dataset.collate)
    target_val_loader = DataLoader(target_val, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    target_test_loader = DataLoader(target_test, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)

    shadow_train_loader = DataLoader(shadow_train, batch_size=params['batch_size'], shuffle=True, collate_fn=dataset.collate)
    shadow_val_loader = DataLoader(shadow_val, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    shadow_test_loader = DataLoader(shadow_test, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)


    # At any point you can hit Ctrl + C to break out of training early.
    print("==============Start Training Target Model==============")
    print("root_ckpt_dir：",root_ckpt_dir)
    t_ckpt_dir, s_ckpt_dir = '',''
    try:
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:
                t.set_description('Epoch %d' % epoch)
                start = time.time()
                epoch_train_loss, epoch_train_acc, t_optimizer = train_epoch(t_model,
                                                                             t_optimizer,
                                                                             device,
                                                                             target_train_loader, epoch)
                epoch_val_loss, epoch_val_acc = evaluate_network(t_model, device, target_val_loader, epoch)
                _, epoch_test_acc = evaluate_network(t_model, device, target_test_loader, epoch)

                t_epoch_train_losses.append(epoch_train_loss)
                t_epoch_val_losses.append(epoch_val_loss)
                t_epoch_train_accs.append(epoch_train_acc)
                t_epoch_val_accs.append(epoch_val_acc)

                writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                writer.add_scalar('train/_acc', epoch_train_acc, epoch)
                writer.add_scalar('val/_acc', epoch_val_acc, epoch)
                writer.add_scalar('test/_acc', epoch_test_acc, epoch)
                writer.add_scalar('learning_rate', t_optimizer.param_groups[0]['lr'], epoch)


                t.set_postfix(time=time.time( ) -start, lr=t_optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                              train_acc=epoch_train_acc, val_acc=epoch_val_acc,
                              test_acc=epoch_test_acc)

                per_epoch_time.append(time.time( ) -start)

                # Saving checkpoint
                t_ckpt_dir = os.path.join(root_ckpt_dir, "T_RUN_")
                if not os.path.exists(t_ckpt_dir):
                    os.makedirs(t_ckpt_dir)
                torch.save(t_model.state_dict(), '{}.pkl'.format(t_ckpt_dir + "/epoch_" + str(epoch)))

                files = glob.glob(t_ckpt_dir + '/*.pkl')
                for file in files:
                    epoch_nb = file.split('_')[-1]
                    epoch_nb = int(epoch_nb.split('.')[0])
                    if epoch_nb < epoch -1:
                        os.remove(file)

                t_scheduler.step(epoch_val_loss)

                if t_optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break

                # Stop training after params['max_time'] hours
                if time.time( ) -t0 > params['max_time' ] *3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break
    except KeyboardInterrupt:
        print('-' * 89)
        print('Target Model Training --- Exiting from training early because of KeyboardInterrupt')
    # Train Shadow Model
    print("==============Start Training Shadow Model==============")
    try:
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:
                t.set_description('Epoch %d' % epoch)
                start = time.time()
                epoch_train_loss, epoch_train_acc, s_optimizer = train_epoch(s_model, s_optimizer, device,
                                                                             shadow_train_loader, epoch)
                epoch_val_loss, epoch_val_acc = evaluate_network(s_model, device, shadow_val_loader, epoch)
                _, epoch_test_acc = evaluate_network(s_model, device, shadow_test_loader, epoch)

                s_epoch_train_losses.append(epoch_train_loss)
                s_epoch_val_losses.append(epoch_val_loss)
                s_epoch_train_accs.append(epoch_train_acc)
                s_epoch_val_accs.append(epoch_val_acc)

                writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                writer.add_scalar('train/_acc', epoch_train_acc, epoch)
                writer.add_scalar('val/_acc', epoch_val_acc, epoch)
                writer.add_scalar('test/_acc', epoch_test_acc, epoch)
                writer.add_scalar('learning_rate', s_optimizer.param_groups[0]['lr'], epoch)

                t.set_postfix(time=time.time() - start, lr=s_optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                              train_acc=epoch_train_acc, val_acc=epoch_val_acc,
                              test_acc=epoch_test_acc)

                per_epoch_time.append(time.time() - start)

                # Saving checkpoint
                s_ckpt_dir = os.path.join(root_ckpt_dir, "S_RUN_")
                if not os.path.exists(s_ckpt_dir):
                    os.makedirs(s_ckpt_dir)
                torch.save(s_model.state_dict(), '{}.pkl'.format(s_ckpt_dir + "/epoch_" + str(epoch)))

                files = glob.glob(s_ckpt_dir + '/*.pkl')
                for file in files:
                    epoch_nb = file.split('_')[-1]
                    epoch_nb = int(epoch_nb.split('.')[0])
                    if epoch_nb < epoch - 1:
                        os.remove(file)

                s_scheduler.step(epoch_val_loss)

                if s_optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break

                # Stop training after params['max_time'] hours
                if time.time() - t0 > params['max_time'] * 3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')

    print("=================Evaluate Target Model Start=================")
    _, t_test_acc = evaluate_network(t_model, device, target_test_loader, '0|T|' + t_ckpt_dir)
    _, t_train_acc = evaluate_network(t_model, device, target_train_loader, '1|T|' + t_ckpt_dir)
    print("Target Test Accuracy: {:.4f}".format(t_test_acc))
    print("Target Train Accuracy: {:.4f}".format(t_train_acc))
    print("Target Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TargetTOTAL TIME TAKEN: {:.4f}s".format(time.time() - t0))
    print("Target AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))
    print("=================Evaluate Shadow Model Start=================")
    _, s_test_acc = evaluate_network(s_model, device, shadow_test_loader, '0|S|' + s_ckpt_dir)
    _, s_train_acc = evaluate_network(s_model, device, shadow_train_loader, '1|S|' + s_ckpt_dir)
    print("Shadow Test Accuracy: {:.4f}".format(s_test_acc))
    print("Shadow Train Accuracy: {:.4f}".format(s_train_acc))
    print("Shadow Convergence Time (Epochs): {:.4f}".format(epoch))
    print("Shadow TOTAL TIME TAKEN: {:.4f}s".format(time.time( ) -t0))
    print("Shadow AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    writer.close()

    """
        Write the results in out_dir/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL TARGET RESULTS\nTARGET TEST ACCURACY: {:.4f}\nTARGET TRAIN ACCURACY: {:.4f}\n\n
    FINAL SHADOW RESULTS\nSHADOW TEST ACCURACY: {:.4f}\nSHADOW TRAIN ACCURACY: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n""" \
                .format(DATASET_NAME, MODEL_NAME, params, net_params, s_model, net_params['total_param'],
                        np.mean(np.array(t_test_acc) ) *100, np.mean(np.array(t_train_acc) ) *100,
                        np.mean(np.array(s_test_acc) ) *100, np.mean(np.array(s_train_acc) ) *100,
                        epoch, (time.time( ) -t0 ) /3600, np.mean(per_epoch_time)))



def collate(samples):
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.tensor(labels)



def main():
    """
        USER CONTROLS
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--kernel', help="Please give a value for kernel")
    parser.add_argument('--n_heads', help="Please give a value for n_heads")
    parser.add_argument('--gated', help="Please give a value for gated")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--layer_norm', help="Please give a value for layer_norm")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--sage_aggregator', help="Please give a value for sage_aggregator")
    parser.add_argument('--data_mode', help="Please give a value for data_mode")
    parser.add_argument('--num_pool', help="Please give a value for num_pool")
    parser.add_argument('--gnn_per_block', help="Please give a value for gnn_per_block")
    parser.add_argument('--embedding_dim', help="Please give a value for embedding_dim")
    parser.add_argument('--pool_ratio', help="Please give a value for pool_ratio")
    parser.add_argument('--linkpred', help="Please give a value for linkpred")
    parser.add_argument('--cat', help="Please give a value for cat")
    parser.add_argument('--self_loop', help="Please give a value for self_loop")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    # dataset = LoadData(DATASET_NAME)
    # 使用CIFAR10SuperPixelDataset
    trainset = CIFAR10SuperPixelDataset(split='train')
    testset = CIFAR10SuperPixelDataset(split='test')
    # CIFAR10SuperPixelDataset没有专门的验证集,从训练集中分出一部分作为验证集
    val_size = int(0.1 * len(trainset))  # 使用10%的训练数据作为验证集
    trainset, valset = torch.utils.data.random_split(trainset, [len(trainset) - val_size, val_size])

    def collate(samples):
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.tensor(labels)

    def reduce_dataset(dataset, fraction=0.2):
        num_samples = int(len(dataset) * fraction)
        return random.sample(list(dataset), num_samples)

    # 减小数据集大小
    trainset = reduce_dataset(trainset)
    valset = reduce_dataset(valset)
    testset = reduce_dataset(testset)

        # 合并所有数据
    all_data = trainset + valset + testset
    total_size = len(all_data)

    # 随机选择500个数据点作为Dtarget
    Dtarget = set(random.sample(range(total_size), 50))

    # 随机选择20%作为Daux
    aux_size = int(total_size * 0.1)
    all_indices = set(range(total_size))
    Daux = set(random.sample(all_indices, aux_size))

    # 剩下的数据用于训练、验证和测试
    remaining_indices = list(all_indices - Daux)
    random.shuffle(remaining_indices)

    # 将剩余数据分为训练、验证和测试集
    train_size = int(0.8 * len(remaining_indices))
    val_size = int(0.1 * len(remaining_indices))
    test_size = len(remaining_indices) - train_size - val_size

    train_indices = remaining_indices[:train_size]
    val_indices = remaining_indices[train_size:train_size+val_size]
    test_indices = remaining_indices[train_size+val_size:]

    # # 随机选择10%作为Daux（非成员数据）
    # aux_size = int(total_size * 0.2)
    # all_indices = set(range(total_size))
    # Daux = set(random.sample(all_indices, aux_size))

    # # 从剩余数据中选择20%作为Dtarget（潜在的成员数据）
    # remaining_indices = list(all_indices - Daux)
    # target_size = int(len(remaining_indices) * 0.4)
    # Dtarget = set(random.sample(remaining_indices, target_size))

    # # 剩下的数据用于训练、验证和测试
    # final_remaining_indices = list(set(remaining_indices) - Dtarget)
    # random.shuffle(final_remaining_indices)

    # # 将剩余数据分为训练、验证和测试集
    # train_size = int(0.8 * len(final_remaining_indices))
    # val_size = int(0.1 * len(final_remaining_indices))
    # test_size = len(final_remaining_indices) - train_size - val_size

    # train_indices = remaining_indices[:train_size]
    # val_indices = remaining_indices[train_size:train_size+val_size]
    # test_indices = remaining_indices[train_size+val_size:]

    # 创建新的数据集
    train_data = [all_data[i] for i in train_indices]
    val_data = [all_data[i] for i in val_indices]
    test_data = [all_data[i] for i in test_indices]

    # 分割为target和shadow
    target_train_set, shadow_train_set = random_split(train_data, [len(train_data) // 2, len(train_data) - len(train_data) // 2])
    target_val_set, shadow_val_set = random_split(val_data, [len(val_data) // 2, len(val_data) - len(val_data) // 2])
    target_test_set, shadow_test_set = random_split(test_data, [len(test_data) // 2, len(test_data) - len(test_data) // 2])

    print(f"Total reduced dataset size: {total_size}")
    print(f"Dtarget size: {len(Dtarget)}")
    print(f"Daux size: {len(Daux)}")
    print(f"Remaining data size: {len(remaining_indices)}")
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}, Test size: {len(test_data)}")
    print(f"Target train size: {len(target_train_set)}, val size: {len(target_val_set)}, test size: {len(target_test_set)}")
    print(f"Shadow train size: {len(shadow_train_set)}, val size: {len(shadow_val_set)}, test size: {len(shadow_test_set)}")

    # 创建一个类似于原来 LoadData 返回的对象
    class DatasetWrapper:
        def __init__(self, train, val, test):
            self.train = train
            self.val = val
            self.test = test
            self.name = "CIFAR10SuperPixel"
            self.collate = collate

    # dataset = DatasetWrapper(trainset, valset, testset)
    dataset = DatasetWrapper(target_train_set + shadow_train_set, 
                             target_val_set + shadow_val_set, 
                             target_test_set + shadow_test_set)

    Dtarget_data = [all_data[i] for i in Dtarget]
    Daux_data = [all_data[i] for i in Daux]

    class TargetAuxDatasetWrapper:
        def __init__(self, target, aux):
            self.target = target
            self.aux = aux
            self.name = "CIFAR10SuperPixel_TargetAux"
            self.collate = collate  # 假设 collate 函数已定义

    target_aux_dataset = TargetAuxDatasetWrapper(Dtarget_data, Daux_data)
    
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)
    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    

    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)
    if args.residual is not None:
        net_params['residual'] = True if args.residual=='True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat=='True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.kernel is not None:
        net_params['kernel'] = int(args.kernel)
    if args.n_heads is not None:
        net_params['n_heads'] = int(args.n_heads)
    if args.gated is not None:
        net_params['gated'] = True if args.gated=='True' else False
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.layer_norm is not None:
        net_params['layer_norm'] = True if args.layer_norm=='True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm=='True' else False
    if args.sage_aggregator is not None:
        net_params['sage_aggregator'] = args.sage_aggregator
    if args.data_mode is not None:
        net_params['data_mode'] = args.data_mode
    if args.num_pool is not None:
        net_params['num_pool'] = int(args.num_pool)
    if args.gnn_per_block is not None:
        net_params['gnn_per_block'] = int(args.gnn_per_block)
    if args.embedding_dim is not None:
        net_params['embedding_dim'] = int(args.embedding_dim)
    if args.pool_ratio is not None:
        net_params['pool_ratio'] = float(args.pool_ratio)
    if args.linkpred is not None:
        net_params['linkpred'] = True if args.linkpred=='True' else False
    if args.cat is not None:
        net_params['cat'] = True if args.cat=='True' else False
    if args.self_loop is not None:
        net_params['self_loop'] = True if args.self_loop=='True' else False

    net_params['edge_feat'] = True

    # Superpixels
    # net_params['in_dim'] = dataset.train[0][0].ndata['feat'][0].size(0)
    # net_params['in_dim_edge'] = dataset.train[0][0].edata['feat'][0].size(0)
    # num_classes = len(np.unique(np.array(dataset.train[:][1])))
    # net_params['n_classes'] = num_classes

    # Superpixels
    net_params['in_dim'] = trainset[0][0].ndata['feat'].shape[1]
    net_params['in_dim_edge'] = trainset[0][0].edata['feat'].shape[1]
    num_classes = 10  # CIFAR10有10个类别
    net_params['n_classes'] = num_classes

    if MODEL_NAME == 'DiffPool':
        # calculate assignment dimension: pool_ratio * largest graph's maximum
        # number of nodes  in the dataset
        max_num_nodes_train = max([dataset.train[i][0].number_of_nodes() for i in range(len(dataset.train))])
        max_num_nodes_test = max([dataset.test[i][0].number_of_nodes() for i in range(len(dataset.test))])
        max_num_node = max(max_num_nodes_train, max_num_nodes_test)
        net_params['assign_dim'] = int(max_num_node * net_params['pool_ratio']) * net_params['batch_size']

    if MODEL_NAME == 'RingGNN':
        num_nodes_train = [dataset.train[i][0].number_of_nodes() for i in range(len(dataset.train))]
        num_nodes_test = [dataset.test[i][0].number_of_nodes() for i in range(len(dataset.test))]
        num_nodes = num_nodes_train + num_nodes_test
        net_params['avg_node_num'] = int(np.ceil(np.mean(num_nodes)))

    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str \
        (config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str \
        (config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str \
        (config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str \
        (config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file
    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')

    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
    train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs, target_aux_dataset)




main()
















