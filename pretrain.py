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
            batch_e = batch_graphs.edata['feat'].float().to(device) if 'feat' in batch_graphs.edata else None
            optimizer.zero_grad()
            out = model(batch_graphs, batch_x, batch_e)  
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