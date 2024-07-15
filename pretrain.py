# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch_geometric.datasets import TUDataset
# from torch_geometric.data import DataLoader
# from torch_geometric.nn import GCNConv
# from torch_geometric.utils import to_dense_adj
# import numpy as np

# class GCN(nn.Module):
#     def __init__(self, num_node_features, hidden_channels):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(num_node_features, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, num_node_features)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = torch.relu(x)
#         x = self.conv2(x, edge_index)
#         return x

# def mask_features(x, mask_rate=0.15):
#     mask = torch.bernoulli(torch.full(x.shape, mask_rate)).bool()
#     masked_x = x.clone()
#     masked_x[mask] = 0
#     return masked_x, mask

# def pretrain(dataset, hidden_channels=64, num_epochs=100, batch_size=32, lr=0.001):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     model = GCN(dataset.num_node_features, hidden_channels).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.MSELoss()

#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0
#         for data in loader:
#             data = data.to(device)
#             optimizer.zero_grad()
            
#             masked_x, mask = mask_features(data.x)
#             out = model(masked_x, data.edge_index)
            
#             loss = criterion(out[mask], data.x[mask])
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item() * data.num_graphs

#         avg_loss = total_loss / len(dataset)
#         print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

#     return model

# if __name__ == "__main__":
#     dataset = TUDataset(root='data/TUDataset', name='DD')
#     pretrained_model = pretrain(dataset)
#     torch.save(pretrained_model.state_dict(), 'pretrained_gcn_dd.pth')
#     print("Pretraining completed. Model saved.")

import torch
import torch.nn.functional as F
from dgl.data import TUDataset
from dgl.dataloading import GraphDataLoader
from nets.TUs_graph_classification.load_net import gnn_model
from tqdm import tqdm
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def pretrain(model, train_loader, device, num_epochs=50, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_acc = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch_graphs, batch_labels in tqdm(train_loader, desc=f'Pretraining Epoch {epoch+1}/{num_epochs}'):
            batch_graphs = batch_graphs.to(device)
            batch_labels = batch_labels.to(device)
            
            # 获取节点特征
            if 'feat' in batch_graphs.ndata:
                batch_x = batch_graphs.ndata['feat'].float().to(device)
            elif 'node_labels' in batch_graphs.ndata:
                batch_x = batch_graphs.ndata['node_labels'].float().to(device)
            else:
                batch_x = torch.ones((batch_graphs.number_of_nodes(), 1), device=device)
            
            optimizer.zero_grad()
            out = model(batch_graphs, batch_x, None)  # 假设模型不使用边特征
            loss = F.cross_entropy(out, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_graphs.batch_size
            pred = out.max(1)[1]
            correct += pred.eq(batch_labels).sum().item()
            total += batch_graphs.batch_size
        
        avg_loss = total_loss / len(train_loader.dataset)
        accuracy = correct / total
        print(f'Pretraining Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
        
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), 'best_pretrained_model.pth')
    
    print(f'Best pretraining accuracy: {best_acc:.4f}')
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = TUDataset(name='ENZYMES')
    
    # 假设 MODEL_NAME 和 net_params 已经在某处定义
    MODEL_NAME = 'GCN'  # 或其他模型名称
    net_params = {
        'in_dim': dataset.graphs[0].ndata['feat'].shape[1],
        'hidden_dim': 64,
        'out_dim': dataset.num_classes,
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
    
    model = gnn_model(MODEL_NAME, net_params)
    model = model.to(device)
    
    pretrain(model, device, dataset)

if __name__ == '__main__':
    main()