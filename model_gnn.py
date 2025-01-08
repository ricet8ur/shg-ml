from torch_geometric.nn import  GINEConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, Set2Set
from gnn.attention import GlobalAttention
import torch.nn.functional as F
import torch
import torch.nn as nn
import logging
import os
from torch_sparse import SparseTensor
LOG_FILENAME="log.txt"
logging.basicConfig(filename=LOG_FILENAME,level=logging.INFO)
logging.info("This message should go to the log file")
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
class GNNNodeEmbedding(torch.nn.Module):

    def __init__(self, num_layer, emb_dim, drop_ratio=0.5, JK="last", residual=False, gnn_type="gin"):
        """GIN Node Embedding Module"""

        super(GNNNodeEmbedding, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        # add residual connection or not
        self.residual = residual
        self.atom_encoder = nn.Linear(92, emb_dim)
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")


        # List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):

            self.convs.append(GINEConv(nn.Sequential(nn.Linear(emb_dim, emb_dim),
                                  nn.ReLU(), nn.Linear(emb_dim, emb_dim)), train_eps=True, edge_dim=50))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr = batched_data.x, batched_data.edge_index, batched_data.edge_attr
        x = x.float()
        edge_attr = edge_attr.float()
        adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_attr)
        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layer):
            h = self.convs[layer](h_list[layer], adj.t())
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            if self.residual:
                h += h_list[layer]
            h_list.append(h)

        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layers + 1):
                node_representation += h_list[layer]
        return node_representation





class GNN_Graph(nn.Module):

    def __init__(self, num_classes=1, num_layer=4, emb_dim=300, residual=False, drop_ratio=0, JK="last",
                 graph_pooling="attention"):
        super(GNN_Graph, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_classes = num_classes

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn_node = GNNNodeEmbedding(num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual,gnn_type="gin")
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool0 = GlobalAttention(gate_nn=nn.Sequential(
                nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(), nn.Linear(emb_dim, 1)))
        elif graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = nn.Linear(2 * self.emb_dim, self.num_classes)
        else:
            self.graph_pred_linear = nn.Linear(self.emb_dim, self.num_classes)
        self.hidden0 = nn.Sequential(
            nn.Linear(self.emb_dim+10, (self.emb_dim+10)* 2),
            nn.ReLU(),
            nn.Linear((self.emb_dim+10)* 2, self.emb_dim+10),
            nn.ReLU(),
            nn.Linear(self.emb_dim+10, self.num_classes))
    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool0(h_node, batched_data.batch, batched_data.ptr)
        gap = batched_data.g
        h = torch.cat((h_graph, gap), dim=1)
        output = self.hidden0(h)
        return output

