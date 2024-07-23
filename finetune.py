import torch
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from nets.TUs_graph_classification.load_net import gnn_model
from tqdm import tqdm
import sys
import os


# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def finetune(model, optimizer, device, train_loader, epoch):

    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    for batch_graphs, batch_labels in tqdm(train_loader, desc=f'Finetuning Epoch {epoch+1}'):
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.to(device)
        
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
        
        epoch_loss += loss.item() * batch_graphs.batch_size
        pred = out.max(1)[1]
        correct += pred.eq(batch_labels).sum().item()
        total += batch_graphs.batch_size
    
    avg_loss = epoch_loss / len(train_loader.dataset)
    accuracy = correct / total
    print(f'Finetuning Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    
    return avg_loss, optimizer

# def main():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     dataset = TUsDataset('DD')  # 使用 DD 数据集进行微调
    
#     # 创建数据加载器
#     train_loader = GraphDataLoader(dataset.train, batch_size=32, shuffle=True, collate_fn=dataset.collate)
    
#     # 假设 MODEL_NAME 和 net_params 已经在某处定义
#     MODEL_NAME = 'GCN'  # 或其他模型名称
#     net_params = {
#         'in_dim': dataset.dataset[0][0].ndata['node_labels'].shape[1] if 'node_labels' in dataset.dataset[0][0].ndata else 1,
#         'hidden_dim': 64,
#         'out_dim': len(set(dataset.dataset.graph_labels)),
#         'n_classes': len(set(dataset.dataset.graph_labels)),
#         'in_feat_dropout': 0.0,
#         'dropout': 0.0,
#         'L': 4,
#         'readout': 'mean',
#         'graph_norm': True,
#         'batch_norm': True,
#         'residual': True,
#         'device': device
#     }
    
#     model = gnn_model(MODEL_NAME, net_params)
#     model = model.to(device)
    
#     # 加载预训练模型
#     model.load_state_dict(torch.load('best_pretrained_model.pth'))
    
#     finetune(model, train_loader, device)

# if __name__ == '__main__':
#     main()