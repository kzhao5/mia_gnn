# import numpy as np
# import dgl
# import os
# import time
# import random
# import glob
# import argparse, json
# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader, random_split

# from tensorboardX import SummaryWriter
# from tqdm import tqdm


# class DotDict(dict):
#     def __init__(self, **kwds):
#         self.update(kwds)
#         self.__dict__ = self


# """
#     IMPORTING CUSTOM MODULES/METHODS
# """

# from nets.TUs_graph_classification.load_net import gnn_model  # import GNNs
# from data.data import LoadData  # import dataset

# """
#     GPU Setup
# """


# # def gpu_setup(use_gpu, gpu_id):
# #     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# #     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

# #     if torch.cuda.is_available() and use_gpu:
# #         print('cuda available with GPU:', torch.cuda.get_device_name(0))
# #         device = torch.device("cuda")
# #     else:
# #         print('cuda not available')
# #         device = torch.device("cpu")
# #     return device

# def gpu_setup(use_gpu, gpu_id):
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

#     print(f"CUDA is available: {torch.cuda.is_available()}")
#     print(f"use_gpu flag: {use_gpu}")
#     print(f"gpu_id: {gpu_id}")
    
#     if torch.cuda.is_available() and use_gpu:
#         num_gpus = torch.cuda.device_count()
#         if gpu_id >= num_gpus:
#             print(f"Warning: gpu_id {gpu_id} is not available. Using gpu_id 0 instead.")
#             gpu_id = 0
#         device = torch.device(f"cuda:{gpu_id}")
#         print(f'Using CUDA with GPU: {torch.cuda.get_device_name(gpu_id)}')
#     else:
#         device = torch.device("cpu")
#         print('Using CPU.')
    
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


# def train_val_pipeline(MODEL_NAME, DATASET_NAME, params, net_params, dirs):
#     t_avg_test_acc, t_avg_train_acc, t_avg_convergence_epochs = [], [], []
#     s_avg_test_acc, s_avg_train_acc, s_avg_convergence_epochs = [], [], []

#     t0 = time.time()
#     per_epoch_time = []

#     dataset = LoadData(DATASET_NAME)

#     if MODEL_NAME in ['GCN', 'GAT']:
#         if net_params['self_loop']:
#             print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
#             dataset._add_self_loops()

#     # Read Train Val Test Data
#     trainset, valset, testset = dataset.train, dataset.val, dataset.test
#     print("With Size: {}====={}====={}".format(len(trainset), len(valset), len(testset)))
#     root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
#     device = net_params['device']

#     # Write the network and optimization hyper-parameters in folder config/
#     with open(write_config_file + '.txt', 'w') as f:
#         f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n""".format(
#             DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))

#     # At any point you can hit Ctrl + C to break out of training early.
#     # try:
#     # for split_number in range(10):
#     split_number = random.randint(0, 9)
#     t0_split = time.time()
#     log_dir = os.path.join(root_log_dir, "RUN_" + str(split_number))
#     writer = SummaryWriter(log_dir=log_dir)

#     # setting seeds
#     random.seed(params['seed'])
#     np.random.seed(params['seed'])
#     torch.manual_seed(params['seed'])
#     if device.type == 'cuda':
#         torch.cuda.manual_seed(params['seed'])

#     print("RUN NUMBER: ", split_number)

#     print("Number of Classes: ", net_params['n_classes'])

#     # Init Target Model
#     t_model = gnn_model(MODEL_NAME, net_params)
#     t_model = t_model.to(device)
#     t_optimizer = optim.Adam(t_model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
#     t_scheduler = optim.lr_scheduler.ReduceLROnPlateau(t_optimizer, mode='min',
#                                                        factor=params['lr_reduce_factor'],
#                                                        patience=params['lr_schedule_patience'],
#                                                        verbose=True)
#     print("Target Model:\n{}".format(t_model))
#     # Init Shadow Model
#     s_model = gnn_model(MODEL_NAME, net_params)
#     s_model = s_model.to(device)
#     s_optimizer = optim.Adam(s_model.parameters(), lr=params['init_lr'],
#                              weight_decay=params['weight_decay'])
#     s_scheduler = optim.lr_scheduler.ReduceLROnPlateau(s_optimizer, mode='min',
#                                                        factor=params['lr_reduce_factor'],
#                                                        patience=params['lr_schedule_patience'],
#                                                        verbose=True)
#     print("Shadow Model:\n{}".format(s_model))
#     t_epoch_train_losses, t_epoch_val_losses, t_epoch_train_accs, t_epoch_val_accs = [], [], [], []
#     s_epoch_train_losses, s_epoch_val_losses, s_epoch_train_accs, s_epoch_val_accs = [], [], [], []


#     # Set train, val and test data size
#     train_size = params['train_size']
#     val_size = params['val_size']
#     test_size = params['test_size']
#     # Load Train, Val and Test Dataset
#     trainset, valset, testset = dataset.train[split_number], dataset.val[split_number], dataset.test[split_number]
#     print("Size of Trainset:{}, Valset:{}, Testset:{}".format(len(trainset), len(valset), len(testset)))
#     print("Needed Train size:{} , Val Size:{} and Test Size：{}".format(train_size, val_size, test_size))

#     # In order to flexible manage the size of Train, Val, Test data,
#     # Here we resize the size
#     dataset_all = trainset + testset + valset
#     trainset, valset, testset = random_split(dataset_all,
#                                                       [len(dataset_all) - val_size * 2 - test_size * 2, val_size * 2,
#                                                        test_size * 2])
#     print("Adjust Size:", len(trainset), len(valset), len(testset))

#     # Split Data into Target and Shadow
#     target_train_set, shadow_train_set = random_split(trainset,[len(trainset) // 2, len(trainset) - len(trainset) // 2])
#     target_val_set, shadow_val_set = random_split(valset, [len(valset) // 2, len(valset) - len(valset) // 2])
#     target_test_set, shadow_test_set = random_split(testset, [len(testset) // 2, len(testset) - len(testset) // 2])

#     print("Target train set with size:{} and Shadow train set with size:{}".format(len(target_train_set),
#                                                                                    len(shadow_train_set)))
#     print("Target Val set with size:{} and Shadow Val set with size:{}".format(len(target_val_set),
#                                                                                len(shadow_val_set)))
#     print("Target Test set with size:{} and Shadow Test set with size:{}".format(len(target_test_set),
#                                                                                  len(shadow_test_set)))

#      # sample defined size of graphs
#     selected_T_train_set, _ = random_split(target_train_set, [train_size, len(target_train_set) - train_size])
#     selected_T_val_set, _ = random_split(target_val_set, [val_size, len(target_val_set) - val_size])
#     selected_T_test_set, _ = random_split(target_test_set, [test_size, len(target_test_set) - test_size])
#     print('Selected Training Size:{}, Validation Size: {} and Testing Size:{}'.format(len(selected_T_train_set),
#                                                                                       len(selected_T_val_set),
#                                                                                       len(selected_T_test_set)))

#     selected_S_train_set, _ = random_split(shadow_train_set, [train_size, len(shadow_train_set) - train_size])
#     selected_S_val_set, _ = random_split(shadow_val_set, [val_size, len(shadow_val_set) - val_size])
#     selected_S_test_set, _ = random_split(shadow_test_set, [test_size, len(shadow_test_set) - test_size])
#     print('Selected Shadow Size:{}, Validation Size: {} and Testing Size:{}'.format(len(selected_S_train_set),
#                                                                                     len(selected_S_val_set),
#                                                                                     len(selected_S_test_set)))

#     # batching exception for Diffpool
#     drop_last = True if MODEL_NAME == 'DiffPool' else False

#     # import train functions for all other GCNs
#     from train.train_TUs_graph_classification import train_epoch_sparse as train_epoch, \
#         evaluate_network_sparse as evaluate_network

#     # Load data
#     target_train_loader = DataLoader(selected_T_train_set, batch_size=params['batch_size'], shuffle=True,
#                                      drop_last=drop_last,
#                                      collate_fn=dataset.collate)
#     target_val_loader = DataLoader(selected_T_val_set, batch_size=params['batch_size'], shuffle=False,
#                                    drop_last=drop_last, collate_fn=dataset.collate)
#     target_test_loader = DataLoader(selected_T_test_set, batch_size=params['batch_size'], shuffle=False,
#                                     drop_last=drop_last, collate_fn=dataset.collate)

#     shadow_train_loader = DataLoader(selected_S_train_set, batch_size=params['batch_size'], shuffle=True,
#                                      drop_last=drop_last, collate_fn=dataset.collate)
#     shadow_val_loader = DataLoader(selected_S_val_set, batch_size=params['batch_size'], shuffle=False,
#                                    drop_last=drop_last, collate_fn=dataset.collate)
#     shadow_test_loader = DataLoader(selected_S_test_set, batch_size=params['batch_size'], shuffle=False,
#                                     drop_last=drop_last, collate_fn=dataset.collate)

#     print('Start Training Target Model...')
#     print("target_train_loader:", len(target_train_loader))
#     t_ckpt_dir, s_ckpt_dir = '', ''
#     try:
#         with tqdm(range(params['epochs'])) as t1:
#             for epoch in t1:

#                 t1.set_description('Epoch %d' % epoch)

#                 start = time.time()
#                 # else:   # for all other models common train function
#                 t_epoch_train_loss, t_epoch_train_acc, t_optimizer = train_epoch(t_model,
#                                                                                  t_optimizer,
#                                                                                  device,
#                                                                                  target_train_loader, epoch)

#                 t_epoch_val_loss, t_epoch_val_acc = evaluate_network(t_model, device, target_val_loader, epoch)
#                 _, t_epoch_test_acc = evaluate_network(t_model, device, target_test_loader, epoch)

#                 t_epoch_train_losses.append(t_epoch_train_loss)
#                 t_epoch_val_losses.append(t_epoch_val_loss)
#                 t_epoch_train_accs.append(t_epoch_train_acc)
#                 t_epoch_val_accs.append(t_epoch_val_acc)

#                 writer.add_scalar('train/_loss', t_epoch_train_loss, epoch)
#                 writer.add_scalar('val/_loss', t_epoch_val_loss, epoch)
#                 writer.add_scalar('train/_acc', t_epoch_train_acc, epoch)
#                 writer.add_scalar('val/_acc', t_epoch_val_acc, epoch)
#                 writer.add_scalar('test/_acc', t_epoch_test_acc, epoch)
#                 writer.add_scalar('learning_rate', t_optimizer.param_groups[0]['lr'], epoch)

#                 _, t_epoch_test_acc = evaluate_network(t_model, device, target_test_loader, epoch)
#                 t1.set_postfix(time=time.time() - start, lr=t_optimizer.param_groups[0]['lr'],
#                                train_loss=t_epoch_train_loss, val_loss=t_epoch_val_loss,
#                                train_acc=t_epoch_train_acc, val_acc=t_epoch_val_acc,
#                                test_acc=t_epoch_test_acc)

#                 per_epoch_time.append(time.time() - start)

#                 # Saving checkpoint
#                 t_ckpt_dir = os.path.join(root_ckpt_dir, "T_RUN_")
#                 if not os.path.exists(t_ckpt_dir):
#                     os.makedirs(t_ckpt_dir)
#                 torch.save(t_model.state_dict(), '{}.pkl'.format(t_ckpt_dir + "/epoch_" + str(epoch)))

#                 files = glob.glob(t_ckpt_dir + '/*.pkl')
#                 for file in files:
#                     epoch_nb = file.split('_')[-1]
#                     epoch_nb = int(epoch_nb.split('.')[0])
#                     if epoch_nb < epoch - 1:
#                         os.remove(file)
#                 '''
#                   Update Params
#                 '''
#                 t_scheduler.step(t_epoch_val_loss)

#                 if t_optimizer.param_groups[0]['lr'] < params['min_lr']:
#                     print("\n!! LR EQUAL TO MIN LR SET.")
#                     break

#                 # Stop training after params['max_time'] hours
#                 if time.time() - t0_split > params[
#                     'max_time'] * 3600:  # Dividing max_time by 10, since there are 10 runs in TUs
#                     print('-' * 89)
#                     print("Max_time for one train experiment elapsed {:.3f} hours, so stopping".format(
#                         params['max_time']))
#                     break

#     except KeyboardInterrupt:
#         print('-' * 89)
#         print('Target Model Training --- Exiting from training early because of KeyboardInterrupt')

#     try:
#         # Start training Shadow Model
#         with tqdm(range(params['epochs'])) as t2:
#             for epoch in t2:
#                 t2.set_description('Epoch %d' % epoch)
#                 start = time.time()
#                 s_epoch_train_loss, s_epoch_train_acc, s_optimizer = train_epoch(s_model, s_optimizer, device,
#                                                                                  shadow_train_loader, epoch)

#                 s_epoch_val_loss, s_epoch_val_acc = evaluate_network(s_model, device, shadow_val_loader, epoch)
#                 _, s_epoch_test_acc = evaluate_network(s_model, device, shadow_test_loader, epoch)

#                 s_epoch_train_losses.append(s_epoch_train_loss)
#                 s_epoch_val_losses.append(s_epoch_val_loss)
#                 s_epoch_train_accs.append(s_epoch_train_acc)
#                 s_epoch_val_accs.append(s_epoch_val_acc)

#                 writer.add_scalar('train/_loss', s_epoch_train_loss, epoch)
#                 writer.add_scalar('val/_loss', s_epoch_val_loss, epoch)
#                 writer.add_scalar('train/_acc', s_epoch_train_acc, epoch)
#                 writer.add_scalar('val/_acc', s_epoch_val_acc, epoch)
#                 writer.add_scalar('test/_acc', s_epoch_test_acc, epoch)
#                 writer.add_scalar('learning_rate', s_optimizer.param_groups[0]['lr'], epoch)

#                 _, s_epoch_test_acc = evaluate_network(s_model, device, shadow_test_loader, epoch)
#                 t2.set_postfix(time=time.time() - start, lr=s_optimizer.param_groups[0]['lr'],
#                                train_loss=s_epoch_train_loss, val_loss=s_epoch_val_loss,
#                                train_acc=s_epoch_train_acc, val_acc=s_epoch_val_acc,
#                                test_acc=s_epoch_test_acc)

#                 per_epoch_time.append(time.time() - start)

#                 # Saving checkpoint
#                 s_ckpt_dir = os.path.join(root_ckpt_dir, "S_RUN_")
#                 if not os.path.exists(s_ckpt_dir):
#                     os.makedirs(s_ckpt_dir)
#                 torch.save(s_model.state_dict(), '{}.pkl'.format(s_ckpt_dir + "/epoch_" + str(epoch)))

#                 files = glob.glob(s_ckpt_dir + '/*.pkl')
#                 for file in files:
#                     epoch_nb = file.split('_')[-1]
#                     epoch_nb = int(epoch_nb.split('.')[0])
#                     if epoch_nb < epoch - 1:
#                         os.remove(file)

#                 s_scheduler.step(s_epoch_val_loss)

#                 if s_optimizer.param_groups[0]['lr'] < params['min_lr']:
#                     print("\n!! LR EQUAL TO MIN LR SET.")
#                     break

#                 # Stop training after params['max_time'] hours
#                 if time.time() - t0_split > params[
#                     'max_time'] * 3600:  # Dividing max_time by 10, since there are 10 runs in TUs
#                     print('-' * 89)
#                     print("Max_time for one train experiment elapsed {:.3f} hours, so stopping".format(
#                         params['max_time']))
#                     break
#     except KeyboardInterrupt:
#         print('-' * 89)
#         print('Shadow Model Training --- Exiting from training early because of KeyboardInterrupt')

#     print("=================Evaluate Target Model Start=================")
#     _, t_test_acc = evaluate_network(t_model, device, target_test_loader, '0|T|' + t_ckpt_dir)
#     _, t_train_acc = evaluate_network(t_model, device, target_train_loader, '1|T|' + t_ckpt_dir)
#     t_avg_test_acc.append(t_test_acc)
#     t_avg_train_acc.append(t_train_acc)
#     t_avg_convergence_epochs.append(epoch)
#     print("Target Test Accuracy [LAST EPOCH]: {:.4f}".format(t_test_acc))
#     print("Target Train Accuracy [LAST EPOCH]: {:.4f}".format(t_train_acc))
#     print("Target Convergence Time (Epochs): {:.4f}".format(epoch))
#     _, s_test_acc = evaluate_network(s_model, device, shadow_test_loader, '0|S|' + s_ckpt_dir)
#     _, s_train_acc = evaluate_network(s_model, device, shadow_train_loader, '1|S|' + s_ckpt_dir)
#     s_avg_test_acc.append(s_test_acc)
#     s_avg_train_acc.append(s_train_acc)
#     s_avg_convergence_epochs.append(epoch)
#     print("Shadow Test Accuracy [LAST EPOCH]: {:.4f}".format(s_test_acc))
#     print("Shadow Train Accuracy [LAST EPOCH]: {:.4f}".format(s_train_acc))
#     print("Shadow Convergence Time (Epochs): {:.4f}".format(epoch))

#     print("TOTAL TIME TAKEN: {:.4f}hrs".format((time.time() - t0) / 3600))
#     print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))
#     print("AVG CONVERGENCE Time (Epochs): {:.4f}".format(np.mean(np.array(s_avg_convergence_epochs))))
#     # Final test accuracy value averaged over 10-fold
#     print("""\n\n\nFINAL RESULTS\n\nTEST ACCURACY averaged: {:.4f} with s.d. {:.4f}"""
#           .format(np.mean(np.array(s_avg_test_acc)) * 100, np.std(s_avg_test_acc) * 100))
#     print("\nAll splits Test Accuracies:\n", s_avg_test_acc)
#     print("""\n\n\nFINAL RESULTS\n\nTRAIN ACCURACY averaged: {:.4f} with s.d. {:.4f}"""
#           .format(np.mean(np.array(s_avg_train_acc)) * 100, np.std(s_avg_train_acc) * 100))
#     print("\nAll splits Train Accuracies:\n", s_avg_train_acc)

#     writer.close()

#     """
#         Write the results in out/results folder
#     """
#     with open(write_file_name + '.txt', 'w') as f:
#         f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
#     FINAL TARGET RESULTS\nTARGET TEST ACCURACY averaged: {:.4f} with s.d. {:.4f}\nTARGET TRAIN ACCURACY averaged: {:.4f} with s.d. {:.4f}\n\n
#     FINAL SHADOW RESULTS\nSHADOW TEST ACCURACY averaged: {:.4f} with s.d. {:.4f}\nSHADOW TRAIN ACCURACY averaged: {:.4f} with s.d. {:.4f}\n\n
#     Average Convergence Time (Epochs): {:.4f} with s.d. {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\nAll Splits Test Accuracies: {}""" \
#                 .format(DATASET_NAME, MODEL_NAME, params, net_params, s_model, net_params['total_param'],
#                         np.mean(np.array(t_avg_test_acc)) * 100, np.std(t_avg_test_acc) * 100,
#                         np.mean(np.array(t_avg_train_acc)) * 100, np.std(t_avg_train_acc) * 100,
#                         np.mean(np.array(s_avg_test_acc)) * 100, np.std(s_avg_test_acc) * 100,
#                         np.mean(np.array(s_avg_train_acc)) * 100, np.std(s_avg_train_acc) * 100,
#                         np.mean(t_avg_convergence_epochs), np.std(t_avg_convergence_epochs),
#                         (time.time() - t0) / 3600, np.mean(per_epoch_time), t_avg_test_acc))


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
#     dataset = LoadData(DATASET_NAME)
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
#         net_params['residual'] = True if args.residual == 'True' else False
#     if args.edge_feat is not None:
#         net_params['edge_feat'] = True if args.edge_feat == 'True' else False
#     if args.readout is not None:
#         net_params['readout'] = args.readout
#     if args.kernel is not None:
#         net_params['kernel'] = int(args.kernel)
#     if args.n_heads is not None:
#         net_params['n_heads'] = int(args.n_heads)
#     if args.gated is not None:
#         net_params['gated'] = True if args.gated == 'True' else False
#     if args.in_feat_dropout is not None:
#         net_params['in_feat_dropout'] = float(args.in_feat_dropout)
#     if args.dropout is not None:
#         net_params['dropout'] = float(args.dropout)
#     if args.layer_norm is not None:
#         net_params['layer_norm'] = True if args.layer_norm == 'True' else False
#     if args.batch_norm is not None:
#         net_params['batch_norm'] = True if args.batch_norm == 'True' else False
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
#         net_params['linkpred'] = True if args.linkpred == 'True' else False
#     if args.cat is not None:
#         net_params['cat'] = True if args.cat == 'True' else False
#     if args.self_loop is not None:
#         net_params['self_loop'] = True if args.self_loop == 'True' else False

#     # TUs
#     net_params['in_dim'] = dataset.all.graph_lists[0].ndata['feat'][0].shape[0]
#     num_classes = len(np.unique(dataset.all.graph_labels))
#     net_params['n_classes'] = num_classes

#     if MODEL_NAME == 'DiffPool':
#         # calculate assignment dimension: pool_ratio * largest graph's maximum
#         # number of nodes  in the dataset
#         num_nodes = [dataset.all[i][0].number_of_nodes() for i in range(len(dataset.all))]
#         max_num_node = max(num_nodes)
#         net_params['assign_dim'] = int(max_num_node * net_params['pool_ratio']) * net_params['batch_size']

#     if MODEL_NAME == 'RingGNN':
#         num_nodes = [dataset.all[i][0].number_of_nodes() for i in range(len(dataset.all))]
#         net_params['avg_node_num'] = int(np.ceil(np.mean(num_nodes)))

#     root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
#         config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
#     root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
#         config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
#     write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
#         config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
#     write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
#         config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
#     dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

#     if not os.path.exists(out_dir + 'results'):
#         os.makedirs(out_dir + 'results')

#     if not os.path.exists(out_dir + 'configs'):
#         os.makedirs(out_dir + 'configs')

#     net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
#     train_val_pipeline(MODEL_NAME, DATASET_NAME, params, net_params, dirs)


# main()

# import numpy as np
# import dgl
# import os
# import time
# import random
# import glob
# import argparse, json
# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader, random_split
# import torch.nn.functional as F
# import pickle

# from tensorboardX import SummaryWriter
# from tqdm import tqdm

# class DotDict(dict):
#     def __init__(self, **kwds):
#         self.update(kwds)
#         self.__dict__ = self

# from nets.TUs_graph_classification.load_net import gnn_model  # import GNNs
# from data.data import LoadData  # import dataset

# def gpu_setup(use_gpu, gpu_id):
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

#     print(f"CUDA is available: {torch.cuda.is_available()}")
#     print(f"use_gpu flag: {use_gpu}")
#     print(f"gpu_id: {gpu_id}")
    
#     if torch.cuda.is_available() and use_gpu:
#         num_gpus = torch.cuda.device_count()
#         if gpu_id >= num_gpus:
#             print(f"Warning: gpu_id {gpu_id} is not available. Using gpu_id 0 instead.")
#             gpu_id = 0
#         device = torch.device(f"cuda:{gpu_id}")
#         print(f'Using CUDA with GPU: {torch.cuda.get_device_name(gpu_id)}')
#     else:
#         device = torch.device("cpu")
#         print('Using CPU.')
    
#     return device

# def view_model_param(MODEL_NAME, net_params):
#     model = gnn_model(MODEL_NAME, net_params)
#     total_param = 0
#     print("MODEL DETAILS:\n")
#     for param in model.parameters():
#         total_param += np.prod(list(param.data.size()))
#     print('MODEL/Total parameters:', MODEL_NAME, total_param)
#     return total_param

# def pretrain(model, loader, optimizer, device):
#     model.train()
#     total_loss = 0
#     for batch_graphs, _ in loader:
#         batch_graphs = batch_graphs.to(device)
        
#         # 获取节点特征
#         if 'feat' in batch_graphs.ndata:
#             node_features = batch_graphs.ndata['feat'].float()
#         elif 'node_labels' in batch_graphs.ndata:
#             node_features = batch_graphs.ndata['node_labels'].float()
#         else:
#             node_features = torch.ones((batch_graphs.number_of_nodes(), 1), device=device)
        
#         # 创建自监督任务的伪标签（例如，图的节点数）
#         pseudo_labels = torch.tensor([g.number_of_nodes() for g in dgl.unbatch(batch_graphs)], device=device).float()
        
#         # 前向传播
#         output = model(batch_graphs, node_features, None, pretrain=True)
        
#         # 计算损失（使用均方误差作为示例）
#         loss = F.mse_loss(output, pseudo_labels)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
    
#     return total_loss / len(loader)

# def finetune(model, loader, optimizer, device):
#     model.train()
#     total_loss = 0
#     correct = 0
#     total = 0
#     for batched_graph, labels in loader:
#         batched_graph = batched_graph.to(device)
#         labels = labels.to(device)
        
#         if 'feat' in batched_graph.ndata:
#             node_features = batched_graph.ndata['feat'].float()
#         elif 'node_labels' in batched_graph.ndata:
#             node_features = batched_graph.ndata['node_labels'].float()
#         else:
#             node_features = torch.ones((batched_graph.number_of_nodes(), 1), device=device)
        
#         # 前向传播
#         output = model(batched_graph, node_features, None)
        
#         # 计算损失
#         loss = F.cross_entropy(output, labels)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         total_loss += loss.item()
#         _, predicted = torch.max(output.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
    
#     accuracy = correct / total
#     return total_loss / len(loader), accuracy

# def train_val_pipeline(MODEL_NAME, DATASET_NAME, params, net_params, dirs):
#     t_avg_test_acc, t_avg_train_acc, t_avg_convergence_epochs = [], [], []
#     s_avg_test_acc, s_avg_train_acc, s_avg_convergence_epochs = [], [], []

#     t0 = time.time()
#     per_epoch_time = []

#     dataset = LoadData(DATASET_NAME)

#     if MODEL_NAME in ['GCN', 'GAT']:
#         if net_params['self_loop']:
#             print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
#             dataset._add_self_loops()

#     trainset, valset, testset = dataset.train, dataset.val, dataset.test
#     print("Original dataset sizes - Train: {}, Val: {}, Test: {}".format(len(trainset), len(valset), len(testset)))

#     root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
#     device = net_params['device']

#     # Write the network and optimization hyper-parameters in folder config/
#     with open(write_config_file + '.txt', 'w') as f:
#         f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n""".format(
#             DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))

#     # setting seeds
#     random.seed(params['seed'])
#     np.random.seed(params['seed'])
#     torch.manual_seed(params['seed'])
#     if device.type == 'cuda':
#         torch.cuda.manual_seed(params['seed'])

#     print("RUN NUMBER: ", params['seed'])
#     print("Number of Classes: ", net_params['n_classes'])

#     # Init Target Model
#     t_model = gnn_model(MODEL_NAME, net_params, pretrain=True)
#     t_model = t_model.to(device)
#     t_optimizer = optim.Adam(t_model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
#     t_scheduler = optim.lr_scheduler.ReduceLROnPlateau(t_optimizer, mode='min',
#                                                        factor=params['lr_reduce_factor'],
#                                                        patience=params['lr_schedule_patience'],
#                                                        verbose=True)
#     print("Target Model:\n{}".format(t_model))

#     # Init Shadow Model
#     s_model = gnn_model(MODEL_NAME, net_params, pretrain=True)
#     s_model = s_model.to(device)
#     s_optimizer = optim.Adam(s_model.parameters(), lr=params['init_lr'],
#                              weight_decay=params['weight_decay'])
#     s_scheduler = optim.lr_scheduler.ReduceLROnPlateau(s_optimizer, mode='min',
#                                                        factor=params['lr_reduce_factor'],
#                                                        patience=params['lr_schedule_patience'],
#                                                        verbose=True)
#     print("Shadow Model:\n{}".format(s_model))

#     t_epoch_train_losses, t_epoch_val_losses, t_epoch_train_accs, t_epoch_val_accs = [], [], [], []
#     s_epoch_train_losses, s_epoch_val_losses, s_epoch_train_accs, s_epoch_val_accs = [], [], [], []

#     # Ensure each set has at least one sample
#     min_size = 1
#     train_size = max(min_size, min(len(trainset), params['train_size']))
#     val_size = max(min_size, min(len(valset), params['val_size']))
#     test_size = max(min_size, min(len(testset), params['test_size']))

#     # Randomly select samples
#     selected_trainset = random.sample(trainset, train_size)
#     selected_valset = random.sample(valset, val_size)
#     selected_testset = random.sample(testset, test_size)

#     # Split selected data into target and shadow parts
#     target_train_set, shadow_train_set = selected_trainset[:len(selected_trainset)//2], selected_trainset[len(selected_trainset)//2:]
#     target_val_set, shadow_val_set = selected_valset[:len(selected_valset)//2], selected_valset[len(selected_valset)//2:]
#     target_test_set, shadow_test_set = selected_testset[:len(selected_testset)//2], selected_testset[len(selected_testset)//2:]

#     print("Selected dataset sizes - Train: {}, Val: {}, Test: {}".format(len(selected_trainset), len(selected_valset), len(selected_testset)))
#     print("Target dataset sizes - Train: {}, Val: {}, Test: {}".format(len(target_train_set), len(target_val_set), len(target_test_set)))
#     print("Shadow dataset sizes - Train: {}, Val: {}, Test: {}".format(len(shadow_train_set), len(shadow_val_set), len(shadow_test_set)))

#     # Check if any dataset is empty
#     if len(target_train_set) == 0 or len(target_val_set) == 0 or len(target_test_set) == 0 or \
#        len(shadow_train_set) == 0 or len(shadow_val_set) == 0 or len(shadow_test_set) == 0:
#         print("Error: One or more datasets are empty. Cannot proceed with training.")
#         return None, None, None, None, None, None

#     # batching exception for Diffpool
#     drop_last = True if MODEL_NAME == 'DiffPool' else False

#     # import train functions for all other GCNs
#     from train.train_TUs_graph_classification import train_epoch_sparse as train_epoch, \
#         evaluate_network_sparse as evaluate_network

#     # Load data
#     def collate_dgl(samples):
#         graphs, labels = map(list, zip(*samples))
#         batched_graph = dgl.batch(graphs)
#         return batched_graph, torch.tensor(labels)

#     target_train_loader = DataLoader(target_train_set, batch_size=params['batch_size'], shuffle=True,
#                                      drop_last=drop_last, collate_fn=collate_dgl)
#     target_val_loader = DataLoader(target_val_set, batch_size=params['batch_size'], shuffle=False,
#                                    drop_last=drop_last, collate_fn=collate_dgl)
#     target_test_loader = DataLoader(target_test_set, batch_size=params['batch_size'], shuffle=False,
#                                     drop_last=drop_last, collate_fn=collate_dgl)

#     shadow_train_loader = DataLoader(shadow_train_set, batch_size=params['batch_size'], shuffle=True,
#                                      drop_last=drop_last, collate_fn=collate_dgl)
#     shadow_val_loader = DataLoader(shadow_val_set, batch_size=params['batch_size'], shuffle=False,
#                                    drop_last=drop_last, collate_fn=collate_dgl)
#     shadow_test_loader = DataLoader(shadow_test_set, batch_size=params['batch_size'], shuffle=False,
#                                     drop_last=drop_last, collate_fn=collate_dgl)

#     # Check if any of the loaders are empty
#     if len(target_train_loader) == 0 or len(target_val_loader) == 0 or len(target_test_loader) == 0 or \
#        len(shadow_train_loader) == 0 or len(shadow_val_loader) == 0 or len(shadow_test_loader) == 0:
#         print("Error: One or more data loaders are empty. Cannot proceed with training.")
#         return None, None, None, None, None, None

#     # 预训练阶段
#     print('Start Pretraining Target Model...')
#     for epoch in range(params['pretrain_epochs']):
#         pretrain_loss = pretrain(t_model, target_train_loader, t_optimizer, device)
#         print(f'Pretrain Epoch {epoch+1}/{params["pretrain_epochs"]}, Loss: {pretrain_loss:.4f}')

#     print('Start Pretraining Shadow Model...')
#     for epoch in range(params['pretrain_epochs']):
#         pretrain_loss = pretrain(s_model, shadow_train_loader, s_optimizer, device)
#         print(f'Pretrain Epoch {epoch+1}/{params["pretrain_epochs"]}, Loss: {pretrain_loss:.4f}')

#     print('Start Fine-tuning Target Model...')
#     print("target_train_loader:", len(target_train_loader))
#     t_ckpt_dir, s_ckpt_dir = '', ''
#     try:
#         with tqdm(range(params['epochs'])) as t1:
#             for epoch in t1:
#                 t1.set_description('Epoch %d' % epoch)
#                 start = time.time()
#                 t_epoch_train_loss, t_epoch_train_acc, t_optimizer = train_epoch(t_model, t_optimizer, device, target_train_loader, epoch)
#                 t_epoch_val_loss, t_epoch_val_acc = evaluate_network(t_model, device, target_val_loader, epoch)
#                 _, t_epoch_test_acc = evaluate_network(t_model, device, target_test_loader, epoch)

#                 t_epoch_train_losses.append(t_epoch_train_loss)
#                 t_epoch_val_losses.append(t_epoch_val_loss)
#                 t_epoch_train_accs.append(t_epoch_train_acc)
#                 t_epoch_val_accs.append(t_epoch_val_acc)

#                 writer = SummaryWriter(log_dir=root_log_dir)
#                 writer.add_scalar('train/_loss', t_epoch_train_loss, epoch)
#                 writer.add_scalar('val/_loss', t_epoch_val_loss, epoch)
#                 writer.add_scalar('train/_acc', t_epoch_train_acc, epoch)
#                 writer.add_scalar('val/_acc', t_epoch_val_acc, epoch)
#                 writer.add_scalar('test/_acc', t_epoch_test_acc, epoch)
#                 writer.add_scalar('learning_rate', t_optimizer.param_groups[0]['lr'], epoch)

#                 t1.set_postfix(time=time.time() - start, lr=t_optimizer.param_groups[0]['lr'],
#                                train_loss=t_epoch_train_loss, val_loss=t_epoch_val_loss,
#                                train_acc=t_epoch_train_acc, val_acc=t_epoch_val_acc,
#                                test_acc=t_epoch_test_acc)

#                 per_epoch_time.append(time.time() - start)

#                 # Saving checkpoint
#                 # Saving checkpoint
#                 t_ckpt_dir = os.path.join(root_ckpt_dir, "T_RUN_")
#                 if not os.path.exists(t_ckpt_dir):
#                     os.makedirs(t_ckpt_dir)
#                 torch.save(t_model.state_dict(), '{}.pkl'.format(t_ckpt_dir + "/epoch_" + str(epoch)))

#                 files = glob.glob(t_ckpt_dir + '/*.pkl')
#                 for file in files:
#                     epoch_nb = file.split('_')[-1]
#                     epoch_nb = int(epoch_nb.split('.')[0])
#                     if epoch_nb < epoch - 1:
#                         os.remove(file)

#                 t_scheduler.step(t_epoch_val_loss)

#                 if t_optimizer.param_groups[0]['lr'] < params['min_lr']:
#                     print("\n!! LR EQUAL TO MIN LR SET.")
#                     break

#                 # Stop training after params['max_time'] hours
#                 if time.time() - t0 > params['max_time'] * 3600:
#                     print('-' * 89)
#                     print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
#                     break

#     except KeyboardInterrupt:
#         print('-' * 89)
#         print('Exiting from training early because of KeyboardInterrupt')

#     print('Start Fine-tuning Shadow Model...')
#     try:
#         with tqdm(range(params['epochs'])) as t:
#             for epoch in t:
#                 t.set_description('Epoch %d' % epoch)
#                 start = time.time()
#                 s_epoch_train_loss, s_epoch_train_acc, s_optimizer = train_epoch(s_model, s_optimizer, device, shadow_train_loader, epoch)
#                 s_epoch_val_loss, s_epoch_val_acc = evaluate_network(s_model, device, shadow_val_loader, epoch)
#                 _, s_epoch_test_acc = evaluate_network(s_model, device, shadow_test_loader, epoch)

#                 s_epoch_train_losses.append(s_epoch_train_loss)
#                 s_epoch_val_losses.append(s_epoch_val_loss)
#                 s_epoch_train_accs.append(s_epoch_train_acc)
#                 s_epoch_val_accs.append(s_epoch_val_acc)

#                 writer.add_scalar('shadow/train/_loss', s_epoch_train_loss, epoch)
#                 writer.add_scalar('shadow/val/_loss', s_epoch_val_loss, epoch)
#                 writer.add_scalar('shadow/train/_acc', s_epoch_train_acc, epoch)
#                 writer.add_scalar('shadow/val/_acc', s_epoch_val_acc, epoch)
#                 writer.add_scalar('shadow/test/_acc', s_epoch_test_acc, epoch)
#                 writer.add_scalar('shadow/learning_rate', s_optimizer.param_groups[0]['lr'], epoch)

#                 t.set_postfix(time=time.time() - start, lr=s_optimizer.param_groups[0]['lr'],
#                               train_loss=s_epoch_train_loss, val_loss=s_epoch_val_loss,
#                               train_acc=s_epoch_train_acc, val_acc=s_epoch_val_acc,
#                               test_acc=s_epoch_test_acc)

#                 per_epoch_time.append(time.time() - start)

#                 # Saving checkpoint
#                 s_ckpt_dir = os.path.join(root_ckpt_dir, "S_RUN_")
#                 if not os.path.exists(s_ckpt_dir):
#                     os.makedirs(s_ckpt_dir)
#                 torch.save(s_model.state_dict(), '{}.pkl'.format(s_ckpt_dir + "/epoch_" + str(epoch)))

#                 files = glob.glob(s_ckpt_dir + '/*.pkl')
#                 for file in files:
#                     epoch_nb = file.split('_')[-1]
#                     epoch_nb = int(epoch_nb.split('.')[0])
#                     if epoch_nb < epoch - 1:
#                         os.remove(file)

#                 s_scheduler.step(s_epoch_val_loss)

#                 if s_optimizer.param_groups[0]['lr'] < params['min_lr']:
#                     print("\n!! LR EQUAL TO MIN LR SET.")
#                     break

#                 # Stop training after params['max_time'] hours
#                 if time.time() - t0 > params['max_time'] * 3600:
#                     print('-' * 89)
#                     print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
#                     break

#     except KeyboardInterrupt:
#         print('-' * 89)
#         print('Exiting from training early because of KeyboardInterrupt')

#     # print("Training complete!")
#     # print("Total time taken: {:.2f}s".format(time.time() - t0))

#     # """
#     #     Collect Target and Shadow Model Training Results
#     # """
#     # t_train_acc, t_val_acc, t_test_acc = t_epoch_train_accs[-1], t_epoch_val_accs[-1], t_epoch_test_acc
#     # s_train_acc, s_val_acc, s_test_acc = s_epoch_train_accs[-1], s_epoch_val_accs[-1], s_epoch_test_acc

#     # t_avg_train_acc.append(t_train_acc)
#     # t_avg_test_acc.append(t_test_acc)
#     # s_avg_train_acc.append(s_train_acc)
#     # s_avg_test_acc.append(s_test_acc)

#     # print("Target: Train: {:.4f}, Test: {:.4f}".format(t_train_acc, t_test_acc))
#     # print("Shadow: Train: {:.4f}, Test: {:.4f}".format(s_train_acc, s_test_acc))

#     # # Final Evaluation
#     # print("Final Target Test Accuracy: {:.4f}".format(t_test_acc))
#     # print("Final Shadow Test Accuracy: {:.4f}".format(s_test_acc))

#     # print("Target Convergence Time (Epochs): {}".format(t_epoch_train_accs.index(max(t_epoch_train_accs))))
#     # print("Shadow Convergence Time (Epochs): {}".format(s_epoch_train_accs.index(max(s_epoch_train_accs))))
#     # print("AVG TIME PER EPOCH: {}".format(sum(per_epoch_time) / len(per_epoch_time)))

#     # # Save final results
#     # with open(write_file_name + '.txt', 'w') as f:
#     #     f.write("""Target Convergence Time (Epochs): {}\n"""
#     #             """Shadow Convergence Time (Epochs): {}\n"""
#     #             """AVG TIME PER EPOCH: {}\n\n"""
#     #             """Target Train Accuracy: {:.4f}\n"""
#     #             """Target Test Accuracy: {:.4f}\n"""
#     #             """Shadow Train Accuracy: {:.4f}\n"""
#     #             """Shadow Test Accuracy: {:.4f}\n\n"""
#     #             """Target Model: {}\n\n"""
#     #             """Shadow Model: {}\n\n"""
#     #             """Params: {}\n\n"""
#     #             """Net Parameters: {}\n\n"""
#     #             """Total Parameters: {}\n""".format(
#     #         t_epoch_train_accs.index(max(t_epoch_train_accs)),
#     #         s_epoch_train_accs.index(max(s_epoch_train_accs)),
#     #         sum(per_epoch_time) / len(per_epoch_time),
#     #         t_train_acc, t_test_acc,
#     #         s_train_acc, s_test_acc,
#     #         t_model,
#     #         s_model,
#     #         params,
#     #         net_params,
#     #         net_params['total_param']
#     #     ))

#     # return t_train_acc, t_test_acc, s_train_acc, s_test_acc, t_ckpt_dir, s_ckpt_dir
#     print("Training complete!")
#     print("Total time taken: {:.2f}s".format(time.time() - t0))

#     """
#         Collect Target and Shadow Model Training Results
#     """
#     t_train_acc, t_val_acc, t_test_acc = t_epoch_train_accs[-1], t_epoch_val_accs[-1], t_epoch_test_acc
#     s_train_acc, s_val_acc, s_test_acc = s_epoch_train_accs[-1], s_epoch_val_accs[-1], s_epoch_test_acc

#     t_avg_train_acc.append(t_train_acc)
#     t_avg_test_acc.append(t_test_acc)
#     s_avg_train_acc.append(s_train_acc)
#     s_avg_test_acc.append(s_test_acc)

#     print("Target: Train: {:.4f}, Test: {:.4f}".format(t_train_acc, t_test_acc))
#     print("Shadow: Train: {:.4f}, Test: {:.4f}".format(s_train_acc, s_test_acc))

#     # Final Evaluation
#     print("Final Target Test Accuracy: {:.4f}".format(t_test_acc))
#     print("Final Shadow Test Accuracy: {:.4f}".format(s_test_acc))

#     print("Target Convergence Time (Epochs): {}".format(t_epoch_train_accs.index(max(t_epoch_train_accs))))
#     print("Shadow Convergence Time (Epochs): {}".format(s_epoch_train_accs.index(max(s_epoch_train_accs))))
#     print("AVG TIME PER EPOCH: {}".format(sum(per_epoch_time) / len(per_epoch_time)))

#     # Save final results
#     with open(write_file_name + '.txt', 'w') as f:
#         f.write("""Target Convergence Time (Epochs): {}\n"""
#                 """Shadow Convergence Time (Epochs): {}\n"""
#                 """AVG TIME PER EPOCH: {}\n\n"""
#                 """Target Train Accuracy: {:.4f}\n"""
#                 """Target Test Accuracy: {:.4f}\n"""
#                 """Shadow Train Accuracy: {:.4f}\n"""
#                 """Shadow Test Accuracy: {:.4f}\n\n"""
#                 """Target Model: {}\n\n"""
#                 """Shadow Model: {}\n\n"""
#                 """Params: {}\n\n"""
#                 """Net Parameters: {}\n\n"""
#                 """Total Parameters: {}\n""".format(
#             t_epoch_train_accs.index(max(t_epoch_train_accs)),
#             s_epoch_train_accs.index(max(s_epoch_train_accs)),
#             sum(per_epoch_time) / len(per_epoch_time),
#             t_train_acc, t_test_acc,
#             s_train_acc, s_test_acc,
#             t_model,
#             s_model,
#             params,
#             net_params,
#             net_params['total_param']
#         ))

#     # 添加保存MIA所需数据的代码
#     def save_data(model, data_loader, run_type, save_dir):
#         model.eval()
#         X_train_in, y_train_in = [], []
#         X_train_out, y_train_out = [], []
#         num_nodes_in, num_edges_in = [], []
#         num_nodes_out, num_edges_out = [], []

#         with torch.no_grad():
#             for batch_graphs, batch_labels in data_loader:
#                 batch_graphs = batch_graphs.to(device)
            
#                 # 获取节点特征
#                 if 'feat' in batch_graphs.ndata:
#                     batch_x = batch_graphs.ndata['feat'].to(device)
#                 else:
#                     batch_x = batch_graphs.in_degrees().float().unsqueeze(1).to(device)
            
#                 # 获取边特征（如果存在）
#                 batch_e = batch_graphs.edata['feat'].to(device) if 'feat' in batch_graphs.edata else None
            
#                 batch_scores = model(batch_graphs, batch_x, batch_e)
#                 batch_probs = F.softmax(batch_scores, dim=1).cpu().numpy()
                
#                 for graph, prob, label in zip(dgl.unbatch(batch_graphs), batch_probs, batch_labels):
#                     if label == 1:
#                         X_train_in.append(prob)
#                         y_train_in.append(label.item())
#                         num_nodes_in.append(graph.number_of_nodes())
#                         num_edges_in.append(graph.number_of_edges())
#                     else:
#                         X_train_out.append(prob)
#                         y_train_out.append(label.item())
#                         num_nodes_out.append(graph.number_of_nodes())
#                         num_edges_out.append(graph.number_of_edges())

#         # Save data using pickle
#         with open(os.path.join(save_dir, f'{run_type}_X_train_Label_1.pickle'), 'wb') as f:
#             pickle.dump(np.array(X_train_in), f)
#         with open(os.path.join(save_dir, f'{run_type}_y_train_Label_1.pickle'), 'wb') as f:
#             pickle.dump(np.array(y_train_in), f)
#         with open(os.path.join(save_dir, f'{run_type}_X_train_Label_0.pickle'), 'wb') as f:
#             pickle.dump(np.array(X_train_out), f)
#         with open(os.path.join(save_dir, f'{run_type}_y_train_Label_0.pickle'), 'wb') as f:
#             pickle.dump(np.array(y_train_out), f)
#         with open(os.path.join(save_dir, f'{run_type}_num_node_1.pickle'), 'wb') as f:
#             pickle.dump(np.array(num_nodes_in), f)
#         with open(os.path.join(save_dir, f'{run_type}_num_node_0.pickle'), 'wb') as f:
#             pickle.dump(np.array(num_nodes_out), f)
#         with open(os.path.join(save_dir, f'{run_type}_num_edge_1.pickle'), 'wb') as f:
#             pickle.dump(np.array(num_edges_in), f)
#         with open(os.path.join(save_dir, f'{run_type}_num_edge_0.pickle'), 'wb') as f:
#             pickle.dump(np.array(num_edges_out), f)

#     print("=================Saving Data for MIA=================")
#     save_data(t_model, target_train_loader, 'T', t_ckpt_dir)
#     save_data(s_model, shadow_train_loader, 'S', s_ckpt_dir)

#     return t_train_acc, t_test_acc, s_train_acc, s_test_acc, t_ckpt_dir, s_ckpt_dir

# def main():
#     """
#     USER CONTROLS
#     """
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--use_pretrained', action='store_true', help="Whether to use pretrained model")
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
#     parser.add_argument('--train_size', help="Please give a value for train_size")
#     parser.add_argument('--val_size', help="Please give a value for val_size")
#     parser.add_argument('--test_size', help="Please give a value for test_size")
#     parser.add_argument('--pretrain_epochs', help="Please give a value for pretrain_epochs")
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
#     dataset = LoadData(DATASET_NAME)
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
#     if args.train_size is not None:
#         params['train_size'] = int(args.train_size)
#     if args.val_size is not None:
#         params['val_size'] = int(args.val_size)
#     if args.test_size is not None:
#         params['test_size'] = int(args.test_size)
#     # 新增: 预训练轮数
#     if args.pretrain_epochs is not None:
#         params['pretrain_epochs'] = int(args.pretrain_epochs)
#     else:
#         params['pretrain_epochs'] = 50  # 默认预训练50轮

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
#         net_params['residual'] = True if args.residual == 'True' else False
#     if args.edge_feat is not None:
#         net_params['edge_feat'] = True if args.edge_feat == 'True' else False
#     if args.readout is not None:
#         net_params['readout'] = args.readout
#     if args.kernel is not None:
#         net_params['kernel'] = int(args.kernel)
#     if args.n_heads is not None:
#         net_params['n_heads'] = int(args.n_heads)
#     if args.gated is not None:
#         net_params['gated'] = True if args.gated == 'True' else False
#     if args.in_feat_dropout is not None:
#         net_params['in_feat_dropout'] = float(args.in_feat_dropout)
#     if args.dropout is not None:
#         net_params['dropout'] = float(args.dropout)
#     if args.layer_norm is not None:
#         net_params['layer_norm'] = True if args.layer_norm == 'True' else False
#     if args.batch_norm is not None:
#         net_params['batch_norm'] = True if args.batch_norm == 'True' else False
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
#         net_params['linkpred'] = True if args.linkpred == 'True' else False
#     if args.cat is not None:
#         net_params['cat'] = True if args.cat == 'True' else False
#     if args.self_loop is not None:
#         net_params['self_loop'] = True if args.self_loop == 'True' else False
        
#     # Printing dataset information
#     print("Dataset structure:")
#     print(type(dataset.train))
#     print(type(dataset.train[0]))
#     print(dir(dataset.train[0]))

#     print("Examining dataset structure:")
#     print(f"First item in train set: {dataset.train[0]}")
#     print(f"Type of first graph: {type(dataset.train[0][0])}")
#     print(f"Type of first label: {type(dataset.train[0][1])}")

#     # Setting up model parameters
#     first_graph = dataset.train[0][0]
#     if 'feat' in first_graph.ndata:
#         net_params['in_dim'] = first_graph.ndata['feat'].shape[1]
#     else:
#         net_params['in_dim'] = 1  # 如果没有节点特征，设置为1

#     if 'feat' in first_graph.edata:
#         net_params['in_dim_edge'] = first_graph.edata['feat'].shape[1]
#     else:
#         net_params['in_dim_edge'] = 1  # 如果没有边特征，设置为1

#     # 计算类别数
#     all_labels = [item[1] for item in dataset.train] + [item[1] for item in dataset.val] + [item[1] for item in dataset.test]
#     num_classes = len(set(all_labels))
#     net_params['n_classes'] = num_classes

#     if MODEL_NAME == 'DiffPool':
#         # calculate assignment dimension: pool_ratio * largest graph's maximum
#         # number of nodes in the dataset
#         max_num_nodes_train = max([data[0].number_of_nodes() for data in dataset.train])
#         max_num_nodes_test = max([data[0].number_of_nodes() for data in dataset.test])
#         max_num_node = max(max_num_nodes_train, max_num_nodes_test)
#         net_params['assign_dim'] = int(max_num_node * net_params['pool_ratio']) * net_params['batch_size']

#     if MODEL_NAME == 'RingGNN':
#         num_nodes_train = [data[0].number_of_nodes() for data in dataset.train]
#         num_nodes_test = [data[0].number_of_nodes() for data in dataset.test]
#         num_nodes = num_nodes_train + num_nodes_test
#         net_params['avg_node_num'] = int(np.ceil(np.mean(num_nodes)))

#     # Setting up directories
#     root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
#         config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
#     root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
#         config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
#     write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
#         config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
#     write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
#         config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
#     dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

#     if not os.path.exists(out_dir + 'results'):
#         os.makedirs(out_dir + 'results')
    
#     if not os.path.exists(out_dir + 'configs'):
#         os.makedirs(out_dir + 'configs')

#     net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
#     print(f"Total dataset size: {len(dataset.dataset)}")
#     print(f"Train set size: {len(dataset.train)}")
#     print(f"Val set size: {len(dataset.val)}")
#     print(f"Test set size: {len(dataset.test)}")

#     # 添加是否使用预训练模型的标志
#     params['use_pretrained'] = args.use_pretrained

#     # 调用训练函数
#     train_val_pipeline(MODEL_NAME, DATASET_NAME, params, net_params, dirs)

# if __name__ == '__main__':
#     main()

import numpy as np
import dgl
import os
import time
import random
import glob
import sys
import argparse, json
import torch
import torch.nn.functional as F
import pickle
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nets.TUs_graph_classification.load_net import gnn_model
from data.data import LoadData
from pretrain import pretrain
from finetune import finetune
from train.train_TUs_graph_classification import evaluate_network_sparse

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

from nets.TUs_graph_classification.load_net import gnn_model  # import GNNs
from data.data import LoadData  # import dataset

def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"CUDA is available: {torch.cuda.is_available()}")
    print(f"use_gpu flag: {use_gpu}")
    print(f"gpu_id: {gpu_id}")
    
    if torch.cuda.is_available() and use_gpu:
        num_gpus = torch.cuda.device_count()
        if gpu_id >= num_gpus:
            print(f"Warning: gpu_id {gpu_id} is not available. Using gpu_id 0 instead.")
            gpu_id = 0
        device = torch.device(f"cuda:{gpu_id}")
        print(f'Using CUDA with GPU: {torch.cuda.get_device_name(gpu_id)}')
    else:
        device = torch.device("cpu")
        print('Using CPU.')
    
    return device

def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param

def train_val_pipeline(MODEL_NAME, DATASET_NAME, params, net_params, dirs):
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    t_ckpt_dir = os.path.join(root_ckpt_dir, 'T_RUN_')
    s_ckpt_dir = os.path.join(root_ckpt_dir, 'S_RUN_')
    
    if not os.path.exists(t_ckpt_dir):
        os.makedirs(t_ckpt_dir)
    if not os.path.exists(s_ckpt_dir):
        os.makedirs(s_ckpt_dir)

    t0 = time.time()
    per_epoch_time = []

    DATASET_NAME = 'ENZYMES' if params['use_pretrained'] else DATASET_NAME
    dataset = LoadData(DATASET_NAME)
    
    if MODEL_NAME in ['GCN', 'GAT']:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            dataset._add_self_loops()

    trainset, valset, testset = dataset.train, dataset.val, dataset.test
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']

    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write(f"Dataset: {DATASET_NAME},\nModel: {MODEL_NAME}\n\nparams={params}\n\nnet_params={net_params}\n\n\nTotal Parameters: {net_params['total_param']}\n\n")

    # Setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])

    print("Training Graphs: ", len(trainset))
    print("Validation Graphs: ", len(valset))
    print("Test Graphs: ", len(testset))
    print("Number of Classes: ", net_params['n_classes'])

    model = gnn_model(MODEL_NAME, net_params)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)

    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_accs, epoch_val_accs = [], []

    # train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, collate_fn=dataset.collate)
    # val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    # test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)

    # Target model data loaders
    train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, collate_fn=dataset.collate)
    val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)

    # Shadow model data loaders
    # 假设我们使用相同的数据集，但进行不同的随机分割
    shadow_dataset = LoadData(DATASET_NAME)
    shadow_trainset, shadow_valset, shadow_testset = shadow_dataset.train, shadow_dataset.val, shadow_dataset.test
    
    shadow_train_loader = DataLoader(shadow_trainset, batch_size=params['batch_size'], shuffle=True, collate_fn=dataset.collate)
    shadow_val_loader = DataLoader(shadow_valset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    shadow_test_loader = DataLoader(shadow_testset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)

    # Pretraining
    if params['use_pretrained']:
        print("--------- Pretraining ---------")
        pretrain(model, train_loader, device, params['pretrain_epochs'])
        print("--------- Pretraining Finished ---------")

    # # Training
    # with tqdm(range(params['epochs'])) as t:
    #     for epoch in t:
    #         t.set_description('Epoch %d' % epoch)

    #         start = time.time()
    #         epoch_train_loss, optimizer = finetune(model, optimizer, device, train_loader, epoch)
    #         epoch_val_loss, epoch_val_acc = evaluate_network_sparse(model, device, val_loader, epoch)

    #         epoch_train_losses.append(epoch_train_loss)
    #         epoch_val_losses.append(epoch_val_loss)
    #         epoch_val_accs.append(epoch_val_acc)

    #         writer = SummaryWriter(log_dir=root_log_dir)
    #         writer.add_scalar('train/_loss', epoch_train_loss, epoch)
    #         writer.add_scalar('val/_loss', epoch_val_loss, epoch)
    #         writer.add_scalar('val/_acc', epoch_val_acc, epoch)
    #         writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

    #         t.set_postfix(time=time.time() - start, lr=optimizer.param_groups[0]['lr'],
    #                       train_loss=epoch_train_loss, val_loss=epoch_val_loss,
    #                       val_acc=epoch_val_acc)

    #         per_epoch_time.append(time.time() - start)

    #         scheduler.step(epoch_val_loss)

    #         if optimizer.param_groups[0]['lr'] < params['min_lr']:
    #             print("\n!! LR EQUAL TO MIN LR SET.")
    #             break

    #         # Stop training after params['max_time'] hours
    #         if time.time() - t0 > params['max_time'] * 3600:
    #             print('-' * 89)
    #             print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
    #             break

    # _, test_acc = evaluate_network_sparse(model, device, test_loader, epoch)
    # _, train_acc = evaluate_network_sparse(model, device, train_loader, epoch)
    # print("Test Accuracy: {:.4f}".format(test_acc))
    # print("Train Accuracy: {:.4f}".format(train_acc))
    # print("Convergence Time (Epochs): {:.4f}".format(epoch))
    # print("TOTAL TIME TAKEN: {:.4f}s".format(time.time() - t0))
    # print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    # writer.close()

    # """
    #     Write the results in out/results folder
    # """
    # with open(write_file_name + '.txt', 'w') as f:
    #     f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    # FINAL RESULTS\nTEST ACCURACY: {:.4f}\nTRAIN ACCURACY: {:.4f}\n\n
    # Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n"""\
    #     .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
    #             test_acc, train_acc, epoch, (time.time() - t0)/3600, np.mean(per_epoch_time)))
    print('Start Fine-tuning Target and Shadow Models...')

    t_model = model  
    s_model = model 

    t_optimizer = optimizer
    s_optimizer = torch.optim.Adam(s_model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])

    t_scheduler = scheduler
    s_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(s_optimizer, mode='min', factor=params['lr_reduce_factor'],
                                                            patience=params['lr_schedule_patience'], verbose=True)

    t_epoch_train_losses, t_epoch_val_losses, t_epoch_train_accs, t_epoch_val_accs = [], [], [], []
    s_epoch_train_losses, s_epoch_val_losses, s_epoch_train_accs, s_epoch_val_accs = [], [], [], []

    with tqdm(range(params['epochs'])) as t:
        for epoch in t:
            t.set_description('Epoch %d' % epoch)

            start = time.time()

            # Target model training
            t_epoch_train_loss, t_optimizer = finetune(t_model, t_optimizer, device, train_loader, epoch)
            t_epoch_val_loss, t_epoch_val_acc = evaluate_network_sparse(t_model, device, val_loader, epoch)

            # Shadow model training
            s_epoch_train_loss, s_optimizer = finetune(s_model, s_optimizer, device, shadow_train_loader, epoch)
            s_epoch_val_loss, s_epoch_val_acc = evaluate_network_sparse(s_model, device, shadow_val_loader, epoch)

            # Record losses and accuracies
            t_epoch_train_losses.append(t_epoch_train_loss)
            t_epoch_val_losses.append(t_epoch_val_loss)
            t_epoch_val_accs.append(t_epoch_val_acc)

            s_epoch_train_losses.append(s_epoch_train_loss)
            s_epoch_val_losses.append(s_epoch_val_loss)
            s_epoch_val_accs.append(s_epoch_val_acc)

            # Log to TensorBoard
            writer = SummaryWriter(log_dir=root_log_dir)
            writer.add_scalar('target/train/_loss', t_epoch_train_loss, epoch)
            writer.add_scalar('target/val/_loss', t_epoch_val_loss, epoch)
            writer.add_scalar('target/val/_acc', t_epoch_val_acc, epoch)
            writer.add_scalar('target/learning_rate', t_optimizer.param_groups[0]['lr'], epoch)

            writer.add_scalar('shadow/train/_loss', s_epoch_train_loss, epoch)
            writer.add_scalar('shadow/val/_loss', s_epoch_val_loss, epoch)
            writer.add_scalar('shadow/val/_acc', s_epoch_val_acc, epoch)
            writer.add_scalar('shadow/learning_rate', s_optimizer.param_groups[0]['lr'], epoch)

            t.set_postfix(time=time.time() - start, t_lr=t_optimizer.param_groups[0]['lr'], s_lr=s_optimizer.param_groups[0]['lr'],
                        t_train_loss=t_epoch_train_loss, t_val_loss=t_epoch_val_loss, t_val_acc=t_epoch_val_acc,
                        s_train_loss=s_epoch_train_loss, s_val_loss=s_epoch_val_loss, s_val_acc=s_epoch_val_acc)

            per_epoch_time.append(time.time() - start)

            # Update schedulers
            t_scheduler.step(t_epoch_val_loss)
            s_scheduler.step(s_epoch_val_loss)

            # Check for early stopping
            if t_optimizer.param_groups[0]['lr'] < params['min_lr'] and s_optimizer.param_groups[0]['lr'] < params['min_lr']:
                print("\n!! LR EQUAL TO MIN LR SET.")
                break

            # Stop training after params['max_time'] hours
            if time.time() - t0 > params['max_time'] * 3600:
                print('-' * 89)
                print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                break

    
    # Final evaluation
    _, t_test_acc = evaluate_network_sparse(t_model, device, test_loader, epoch)
    _, t_train_acc = evaluate_network_sparse(t_model, device, train_loader, epoch)

    _, s_test_acc = evaluate_network_sparse(s_model, device, shadow_test_loader, epoch)
    _, s_train_acc = evaluate_network_sparse(s_model, device, shadow_train_loader, epoch)

    print("Target Model - Test Accuracy: {:.4f}, Train Accuracy: {:.4f}".format(t_test_acc, t_train_acc))
    print("Shadow Model - Test Accuracy: {:.4f}, Train Accuracy: {:.4f}".format(s_test_acc, s_train_acc))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time() - t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    writer.close()

    print("Saving models...")
    torch.save(t_model.state_dict(), os.path.join(t_ckpt_dir, 'target_model.pth'))
    torch.save(s_model.state_dict(), os.path.join(s_ckpt_dir, 'shadow_model.pth'))

    # 保存 MIA 所需的数据
    print("=================Saving Data for MIA=================")
    save_data(t_model, train_loader, 'T', t_ckpt_dir, device)
    save_data(s_model, shadow_train_loader, 'S', s_ckpt_dir, device)

    return t_train_acc, t_test_acc, s_train_acc, s_test_acc, t_ckpt_dir, s_ckpt_dir

    # # Write results to file
    # with open(write_file_name + '.txt', 'w') as f:
    #     f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    # FINAL RESULTS\nTARGET MODEL - TEST ACCURACY: {:.4f}, TRAIN ACCURACY: {:.4f}\n
    # SHADOW MODEL - TEST ACCURACY: {:.4f}, TRAIN ACCURACY: {:.4f}\n\n
    # Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n"""\
    #     .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
    #             t_test_acc, t_train_acc, s_test_acc, s_train_acc, epoch, (time.time() - t0)/3600, np.mean(per_epoch_time)))
def save_data(model, data_loader, run_type, save_dir, device):
    model.eval()
    X_train_in, y_train_in = [], []
    X_train_out, y_train_out = [], []
    num_nodes_in, num_edges_in = [], []
    num_nodes_out, num_edges_out = [], []

    with torch.no_grad():
        for batch_graphs, batch_labels in data_loader:
            batch_graphs = batch_graphs.to(device)
            
            # 获取节点特征
            if 'feat' in batch_graphs.ndata:
                batch_x = batch_graphs.ndata['feat'].to(device)
            elif 'node_labels' in batch_graphs.ndata:
                batch_x = batch_graphs.ndata['node_labels'].float().to(device)
            else:
                batch_x = batch_graphs.in_degrees().float().unsqueeze(1).to(device)
            
            # 获取边特征（如果存在）
            batch_e = batch_graphs.edata['feat'].to(device) if 'feat' in batch_graphs.edata else None
            
            batch_scores = model(batch_graphs, batch_x, batch_e)
            batch_probs = F.softmax(batch_scores, dim=1).cpu().numpy()
            
            for graph, prob, label in zip(dgl.unbatch(batch_graphs), batch_probs, batch_labels):
                if label == 1:
                    X_train_in.append(prob)
                    y_train_in.append(label.item())
                    num_nodes_in.append(graph.number_of_nodes())
                    num_edges_in.append(graph.number_of_edges())
                else:
                    X_train_out.append(prob)
                    y_train_out.append(label.item())
                    num_nodes_out.append(graph.number_of_nodes())
                    num_edges_out.append(graph.number_of_edges())

    

    # Save data using pickle
    with open(os.path.join(save_dir, f'{run_type}_X_train_Label_1.pickle'), 'wb') as f:
        pickle.dump(np.array(X_train_in), f)
    with open(os.path.join(save_dir, f'{run_type}_y_train_Label_1.pickle'), 'wb') as f:
        pickle.dump(np.array(y_train_in), f)
    with open(os.path.join(save_dir, f'{run_type}_X_train_Label_0.pickle'), 'wb') as f:
        pickle.dump(np.array(X_train_out), f)
    with open(os.path.join(save_dir, f'{run_type}_y_train_Label_0.pickle'), 'wb') as f:
        pickle.dump(np.array(y_train_out), f)
    with open(os.path.join(save_dir, f'{run_type}_num_node_1.pickle'), 'wb') as f:
        pickle.dump(np.array(num_nodes_in), f)
    with open(os.path.join(save_dir, f'{run_type}_num_node_0.pickle'), 'wb') as f:
        pickle.dump(np.array(num_nodes_out), f)
    with open(os.path.join(save_dir, f'{run_type}_num_edge_1.pickle'), 'wb') as f:
        pickle.dump(np.array(num_edges_in), f)
    with open(os.path.join(save_dir, f'{run_type}_num_edge_0.pickle'), 'wb') as f:
        pickle.dump(np.array(num_edges_out), f)

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
    parser.add_argument('--use_pretrained', action='store_true', help="Whether to use pretrained model")
    parser.add_argument('--pretrain_epochs', help="Please give a value for pretrain_epochs")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
        
    # Device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    
    # Model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    dataset = LoadData(DATASET_NAME)
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    
    # Parameters
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
    if args.use_pretrained:
        params['use_pretrained'] = True
    if args.pretrain_epochs is not None:
        params['pretrain_epochs'] = int(args.pretrain_epochs)
    
    # Network parameters
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
        
    # # Superpixels
    # net_params['in_dim'] = dataset.train[0][0].ndata['feat'][0].size(0)
    # net_params['in_dim_edge'] = dataset.train[0][0].edata['feat'][0].size(0)
    # num_classes = len(np.unique(np.array(dataset.train[:][1])))
    # net_params['n_classes'] = num_classes

    # Setting up model parameters
    first_graph = dataset.train[0][0]
    print("Node feature keys:", first_graph.ndata.keys())
    print("Edge feature keys:", first_graph.edata.keys())

    if len(first_graph.ndata) > 0:
        print("First node feature shape:", first_graph.ndata[list(first_graph.ndata.keys())[0]].shape)
    if len(first_graph.edata) > 0:
        print("First edge feature shape:", first_graph.edata[list(first_graph.edata.keys())[0]].shape)
    
    # 处理节点特征
    if 'node_labels' in first_graph.ndata:
        net_params['in_dim'] = first_graph.ndata['node_labels'].shape[-1]
    else:
        print("Warning: No node features found. Using constant input feature.")
        net_params['in_dim'] = 1

    # 处理边特征（如果需要的话）
    if 'edge_labels' in first_graph.edata:
        net_params['in_dim_edge'] = first_graph.edata['edge_labels'].shape[-1]
    else:
        print("Warning: No edge features found. Using constant edge feature.")
        net_params['in_dim_edge'] = 1

    # 计算类别数
    all_labels = [item[1] for item in dataset.train] + [item[1] for item in dataset.val] + [item[1] for item in dataset.test]
    num_classes = len(set(all_labels))
    net_params['n_classes'] = num_classes

    # 打印特征信息
    print(f"Node feature dimension: {net_params['in_dim']}")
    print(f"Edge feature dimension: {net_params['in_dim_edge']}")
    print(f"Number of classes: {net_params['n_classes']}")

    if MODEL_NAME == 'DiffPool':
        # Calculate assignment dimension: pool_ratio * largest graph's maximum
        # number of nodes  in the dataset
        max_num_nodes = max([dataset.train[i][0].number_of_nodes() for i in range(len(dataset.train))])
        net_params['assign_dim'] = int(max_num_nodes * net_params['pool_ratio']) * net_params['batch_size']
        
    if MODEL_NAME == 'RingGNN':
        num_nodes = [dataset.train[i][0].number_of_nodes() for i in range(len(dataset.train))]
        net_params['avg_node_num'] = int(np.ceil(np.mean(num_nodes)))
    
    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    


    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')
        
    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
    # train_val_pipeline(MODEL_NAME, DATASET_NAME, params, net_params, dirs)
    t_train_acc, t_test_acc, s_train_acc, s_test_acc, t_ckpt_dir, s_ckpt_dir = train_val_pipeline(MODEL_NAME, DATASET_NAME, params, net_params, dirs)

    print(f"Target Model - Train Accuracy: {t_train_acc:.4f}, Test Accuracy: {t_test_acc:.4f}")
    print(f"Shadow Model - Train Accuracy: {s_train_acc:.4f}, Test Accuracy: {s_test_acc:.4f}")
    print(f"Target Model Checkpoint Dir: {t_ckpt_dir}")
    print(f"Shadow Model Checkpoint Dir: {s_ckpt_dir}")
    
if __name__ == '__main__':
    main()
