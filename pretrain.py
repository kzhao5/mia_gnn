import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj
import numpy as np

class GCN(nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_node_features)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

def mask_features(x, mask_rate=0.15):
    mask = torch.bernoulli(torch.full(x.shape, mask_rate)).bool()
    masked_x = x.clone()
    masked_x[mask] = 0
    return masked_x, mask

def pretrain(dataset, hidden_channels=64, num_epochs=100, batch_size=32, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = GCN(dataset.num_node_features, hidden_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            masked_x, mask = mask_features(data.x)
            out = model(masked_x, data.edge_index)
            
            loss = criterion(out[mask], data.x[mask])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * data.num_graphs

        avg_loss = total_loss / len(dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

    return model

if __name__ == "__main__":
    dataset = TUDataset(root='data/TUDataset', name='DD')
    pretrained_model = pretrain(dataset)
    torch.save(pretrained_model.state_dict(), 'pretrained_gcn_dd.pth')
    print("Pretraining completed. Model saved.")