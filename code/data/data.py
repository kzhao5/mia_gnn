# """
#     File to load dataset based on user control from main file
# """
# import torch
# from data.superpixels import SuperPixDataset
# from data.TUs import TUsDataset


# def LoadData(DATASET_NAME):
#     """
#         This function is called in the main.py file 
#         returns:
#         ; dataset object
#     """
#     # Handling for MNIST or CIFAR Superpixels
#     if DATASET_NAME == 'MNIST' or DATASET_NAME == 'CIFAR10':
#         return SuperPixDataset(DATASET_NAME)

#     # Handling for the TU Datasets
#     TU_DATASETS = ['ENZYMES', 'DD', 'PROTEINS_full']
#     if DATASET_NAME in TU_DATASETS: 
#         return TUsDataset(DATASET_NAME)
"""
    File to load dataset based on user control from main file
"""
import torch
from data.superpixels import SuperPixDataset
from data.TUs import TUsDataset
from dgl.data import TUDataset
from dgl.dataloading import GraphDataLoader
import dgl

class DGLFormDataset(object):
    def __init__(self, graphs, labels):
        self.graph_lists = graphs
        self.graph_labels = labels

    def __getitem__(self, idx):
        return self.graph_lists[idx], self.graph_labels[idx]

    def __len__(self):
        return len(self.graph_lists)

class TUsDataset(torch.utils.data.Dataset):
    def __init__(self, name):
        self.name = name
        # self.dataset = TUDataset(name, raw_dir='/tmp/'+name, force_reload=True)
        data_dir = '/home/kzhao/.dgl'  # 使用已下载的数据集路径
        self.dataset = TUDataset(name, raw_dir=data_dir)
        self.train, self.val, self.test = self._split_data()

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def _split_data(self):
        num_graphs = len(self.dataset)
        train_ratio, val_ratio = 0.8, 0.1
        train_size = max(int(num_graphs * train_ratio), 1)  # 确保至少有1个训练样本
        val_size = max(int(num_graphs * val_ratio), 1)  # 确保至少有1个验证样本
        test_size = num_graphs - train_size - val_size
    
        # 如果测试集为空，从训练集和验证集各取一个样本
        if test_size == 0:
            train_size -= 1
            val_size -= 1
            test_size = 2
    
        # Random split
        indices = torch.randperm(num_graphs)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]
    
        train_data = [self.dataset[i] for i in train_indices]
        val_data = [self.dataset[i] for i in val_indices]
        test_data = [self.dataset[i] for i in test_indices]
    
        return train_data, val_data, test_data

    def collate(self, samples):
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.tensor(labels)

def LoadData(DATASET_NAME):
    """
        This function is called in the main.py file 
        returns:
        ; dataset object
    """
    dataset = TUDataset(DATASET_NAME)
    # Handling for MNIST or CIFAR Superpixels
    if DATASET_NAME == 'MNIST' or DATASET_NAME == 'CIFAR10':
        return SuperPixDataset(DATASET_NAME)

    # Handling for the TU Datasets
    TU_DATASETS = ['ENZYMES', 'DD', 'PROTEINS_full']
    if DATASET_NAME in TU_DATASETS: 
        return TUsDataset(DATASET_NAME)
    
    if DATASET_NAME == 'DD':
        # 将标签映射到0和1
        for split in [dataset.train, dataset.val, dataset.test]:
            for i in range(len(split)):
                graph, label = split[i]
                split[i] = (graph, int(label > 0))  # 假设原始标签大于0表示正类
    
    return dataset