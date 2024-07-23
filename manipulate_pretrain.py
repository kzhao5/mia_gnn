import torch
import torch.nn.functional as F
from dgl.data import TUDataset
from dgl.dataloading import GraphDataLoader
from nets.TUs_graph_classification.load_net import gnn_model
from tqdm import tqdm
import random
import sys
import os
import dgl

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def manipulate_pretrain(model, train_loader, device, num_epochs=50, lr=0.01, alpha=0.5, dtarget_percentage=0.15):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 选择 Dtarget
    all_data = list(train_loader.dataset)
    n_target = int(len(all_data) * dtarget_percentage)
    Dtarget = set(random.sample(all_data, n_target))
    
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
            
            # 计算 Daux 和 Dtarget 的损失
            loss_aux = 0
            loss_target = 0
            n_aux = 0
            n_target = 0
            
            for i, (graph, label) in enumerate(zip(dgl.unbatch(batch_graphs), batch_labels)):
                sample = (graph, label.item())
                if sample in Dtarget:
                    loss_target += F.cross_entropy(out[i:i+1], batch_labels[i:i+1])
                    n_target += 1
                else:
                    loss_aux += F.cross_entropy(out[i:i+1], batch_labels[i:i+1])
                    n_aux += 1
            
            # 避免除以零
            n_aux = max(n_aux, 1)
            n_target = max(n_target, 1)
            
            # 计算操纵后的损失
            loss = (alpha * loss_aux / n_aux) - ((1 - alpha) * loss_target / n_target)
            
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
            torch.save(model.state_dict(), 'best_manipulated_pretrained_model.pth')
    
    print(f'Best pretraining accuracy: {best_acc:.4f}')
    return model