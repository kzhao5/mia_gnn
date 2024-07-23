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
from dgl.data import TUDataset
from torch.utils.data import DataLoader
# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def manipulate(model, data, device, num_epochs=100, lr=0.01, alpha=0.5, n_target=1000, aux_ratio=0.1, batch_size=32):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 检查输入是否为 DataLoader，如果不是，创建一个
    if isinstance(data, DataLoader):
        manipulate_loader = data
        dataset_size = len(data.dataset)
    else:
        manipulate_loader = DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=data.collate)
        dataset_size = len(data)
    
    # 创建索引列表
    all_indices = list(range(dataset_size))
    
    # 随机选择n_target个索引作为Dtarget
    n_target = min(n_target, dataset_size)
    Dtarget = set(random.sample(all_indices, n_target))
    
    # 从剩余的索引中选择aux_ratio比例作为Daux
    remaining_indices = [i for i in all_indices if i not in Dtarget]
    n_aux = int(len(remaining_indices) * aux_ratio)
    Daux = set(random.sample(remaining_indices, n_aux))
    
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (batch_graphs, batch_labels) in enumerate(tqdm(manipulate_loader, desc=f'Manipulating Epoch {epoch+1}/{num_epochs}')):
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
            
            start_idx = batch_idx * manipulate_loader.batch_size
            for i in range(len(batch_labels)):
                global_idx = start_idx + i
                if global_idx in Dtarget:
                    loss_target += F.cross_entropy(out[i:i+1], batch_labels[i:i+1])
                    n_target += 1
                elif global_idx in Daux:
                    loss_aux += F.cross_entropy(out[i:i+1], batch_labels[i:i+1])
                    n_aux += 1
            
            # 避免除以零
            n_aux = max(n_aux, 1)
            n_target = max(n_target, 1)
            
            # 计算操纵后的损失
            loss = (alpha * loss_aux / n_aux) + ((1 - alpha) * loss_target / n_target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(manipulate_loader)
        print(f'Manipulating Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'manipulated_pretrained_model.pth')
    
    print(f'Best manipulation loss: {best_loss:.4f}')
    return model
