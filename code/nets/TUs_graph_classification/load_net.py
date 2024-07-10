# """
#     Utility file to select GraphNN model as
#     selected by the user
# """

# from nets.TUs_graph_classification.gated_gcn_net import GatedGCNNet
# from nets.TUs_graph_classification.gcn_net import GCNNet
# from nets.TUs_graph_classification.gat_net import GATNet
# from nets.TUs_graph_classification.graphsage_net import GraphSageNet
# from nets.TUs_graph_classification.gin_net import GINNet
# from nets.TUs_graph_classification.mo_net import MoNet as MoNet_
# from nets.TUs_graph_classification.mlp_net import MLPNet
# from nets.TUs_graph_classification.ring_gnn_net import RingGNNNet
# from nets.TUs_graph_classification.three_wl_gnn_net import ThreeWLGNNNet


# def GatedGCN(net_params):
#     return GatedGCNNet(net_params)


# def GCN(net_params):
#     return GCNNet(net_params)


# def GAT(net_params):
#     return GATNet(net_params)


# def GraphSage(net_params):
#     return GraphSageNet(net_params)


# def GIN(net_params):
#     return GINNet(net_params)


# def MoNet(net_params):
#     return MoNet_(net_params)


# def MLP(net_params):
#     return MLPNet(net_params)


# def RingGNN(net_params):
#     return RingGNNNet(net_params)


# def ThreeWLGNN(net_params):
#     return ThreeWLGNNNet(net_params)


# def gnn_model(MODEL_NAME, net_params):
#     models = {
#         'GatedGCN': GatedGCN,
#         'GCN': GCN,
#         'GAT': GAT,
#         'GraphSage': GraphSage,
#         'GIN': GIN,
#         'MoNet': MoNet_,
#         'MLP': MLP,
#         'RingGNN': RingGNN,
#         '3WLGNN': ThreeWLGNN
#     }

#     return models[MODEL_NAME](net_params)
"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.TUs_graph_classification.gated_gcn_net import GatedGCNNet
from nets.TUs_graph_classification.gcn_net import GCNNet
from nets.TUs_graph_classification.gat_net import GATNet
from nets.TUs_graph_classification.graphsage_net import GraphSageNet
from nets.TUs_graph_classification.gin_net import GINNet
from nets.TUs_graph_classification.mo_net import MoNet as MoNet_
from nets.TUs_graph_classification.mlp_net import MLPNet
from nets.TUs_graph_classification.ring_gnn_net import RingGNNNet
from nets.TUs_graph_classification.three_wl_gnn_net import ThreeWLGNNNet

import torch.nn as nn
import torch.nn.functional as F

class PretrainWrapper(nn.Module):
    def __init__(self, base_model, pretrain_dim):
        super(PretrainWrapper, self).__init__()
        self.base_model = base_model
        self.pretrain_head = nn.Linear(base_model.output_dim, pretrain_dim)
        self.finetune_head = base_model.MLP_layer

    def forward(self, g, h, e, pretrain=False):
        h = self.base_model.forward_graph(g, h, e)
        if pretrain:
            return self.pretrain_head(h).squeeze()
        else:
            return self.finetune_head(h)

    def loss(self, pred, label):
        # 使用交叉熵损失，因为这是一个分类任务
        return F.cross_entropy(pred, label)

def GatedGCN(net_params, pretrain=False):
    base_model = GatedGCNNet(net_params)
    return PretrainWrapper(base_model, 1) if pretrain else base_model

def GCN(net_params, pretrain=False):
    base_model = GCNNet(net_params)
    return PretrainWrapper(base_model, 1) if pretrain else base_model

def GAT(net_params, pretrain=False):
    base_model = GATNet(net_params)
    return PretrainWrapper(base_model, 1) if pretrain else base_model

def GraphSage(net_params, pretrain=False):
    base_model = GraphSageNet(net_params)
    return PretrainWrapper(base_model, 1) if pretrain else base_model

def GIN(net_params, pretrain=False):
    base_model = GINNet(net_params)
    return PretrainWrapper(base_model, 1) if pretrain else base_model

def MoNet(net_params, pretrain=False):
    base_model = MoNet_(net_params)
    return PretrainWrapper(base_model, 1) if pretrain else base_model

def MLP(net_params, pretrain=False):
    base_model = MLPNet(net_params)
    return PretrainWrapper(base_model, 1) if pretrain else base_model

def RingGNN(net_params, pretrain=False):
    base_model = RingGNNNet(net_params)
    return PretrainWrapper(base_model, 1) if pretrain else base_model

def ThreeWLGNN(net_params, pretrain=False):
    base_model = ThreeWLGNNNet(net_params)
    return PretrainWrapper(base_model, 1) if pretrain else base_model

def gnn_model(MODEL_NAME, net_params, pretrain=False):
    models = {
        'GatedGCN': GatedGCN,
        'GCN': GCN,
        'GAT': GAT,
        'GraphSage': GraphSage,
        'GIN': GIN,
        'MoNet': MoNet,
        'MLP': MLP,
        'RingGNN': RingGNN,
        '3WLGNN': ThreeWLGNN
    }

    return models[MODEL_NAME](net_params, pretrain)