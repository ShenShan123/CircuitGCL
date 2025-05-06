import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from torch_geometric.nn import (
    GCNConv, SAGEConv, GATConv, ResGatedGraphConv, 
    GINEConv, ClusterGCNConv
)
from torch_geometric.nn.models.mlp import MLP

NET = 0
DEV = 1
PIN = 2

class GraphHead(nn.Module):
    """ GNN head for graph-level prediction.

    Implementation adapted from the transductive GraphGPS.

    Args:
        hidden_dim (int): Hidden features' dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
        num_layers (int): Number of layers of GNN model
        layers_post_mp (int): number of layers of head MLP
        use_bn (bool): whether to use batch normalization
        drop_out (float): dropout rate
        activation (str): activation function
        src_dst_agg (str): the way to aggregate src and dst nodes, which can be 'concat' or 'add' or 'pool'
    """
    def __init__(self, args):
        super().__init__()
        self.use_cl = args.sgrl
        self.use_stats = args.use_stats
        hidden_dim = args.hid_dim
        node_embed_dim = hidden_dim
        self.task = args.task
        self.task_level = args.task_level
        self.net_only = args.net_only
        self.num_classes = args.num_classes
        self.class_boundaries = args.class_boundaries

        
        ## circuit statistics encoder + PE encoder + node&edge type encoders
        if args.use_stats + self.use_cl == 2:
            assert hidden_dim % 3 == 0, \
                "hidden_dim should be divided by 3 (3 types of encoders)"
            node_embed_dim = hidden_dim // 3

        ## circuit statistics/pe encoder + node&edge type encoders
        elif self.use_stats + self.use_cl == 1:
            assert hidden_dim % 2 == 0, \
                "hidden_dim should be divided by 2 (2 types of encoders)"
            node_embed_dim = hidden_dim // 2

        ## only use node&edge type encoders
        else:
            pass

        ## Contrastive learning encoder
        if self.use_cl:
            self.cl_linear = nn.Linear(args.cl_hid_dim, node_embed_dim)

        ## Circuit Statistics encoder, producing matrix C
        if self.use_stats:
            ## add node_attr transform layer for net/device/pin nodes, by shan
            self.net_attr_layers = nn.Linear(17, node_embed_dim, bias=True)
            self.dev_attr_layers = nn.Linear(17, node_embed_dim, bias=True)
            ## pin attributes are {0, 1, 2} for gate pin, source/drain pin, and base pin
            self.pin_attr_layers = nn.Embedding(17, node_embed_dim)
            self.c_embed_dim = node_embed_dim

        ## Node / Edge type encoders.
        ## Node attributes are {0, 1, 2} for net, device, and pin
        self.node_encoder = nn.Embedding(num_embeddings=4,
                                         embedding_dim=node_embed_dim)
        ## Edge attributes are {0, 1} for 'device-pin' and 'pin-net' edges
        self.edge_encoder = nn.Embedding(num_embeddings=4,
                                         embedding_dim=hidden_dim)
        
        # GNN layers
        self.layers = nn.ModuleList()
        self.model = args.model

        for _ in range(args.num_gnn_layers):
            ## the following are examples of using different GNN layers
            if args.model == 'clustergcn':
                self.layers.append(ClusterGCNConv(hidden_dim, hidden_dim))
            elif args.model == 'gcn':
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
            elif args.model == 'sage':
                self.layers.append(SAGEConv(hidden_dim, hidden_dim))
            elif args.model == 'gat':
                self.layers.append(GATConv(hidden_dim, hidden_dim, heads=1))
            elif args.model == 'resgatedgcn':
                self.layers.append(ResGatedGraphConv(hidden_dim, hidden_dim, edge_dim=hidden_dim))
            elif args.model == 'gine':
                mlp = MLP(
                    in_channels=hidden_dim, 
                    hidden_channels=hidden_dim, 
                    out_channels=hidden_dim, 
                    num_layers=2, 
                    norm=None,
                )
                self.layers.append(GINEConv(mlp, train_eps=True, edge_dim=hidden_dim))
            else:
                raise ValueError(f'Unsupported GNN model: {args.model}')
        
        self.src_dst_agg = args.src_dst_agg

        ## Add graph pooling layer
        if args.src_dst_agg == 'pooladd':
            self.pooling_fun = pygnn.pool.global_add_pool
        elif args.src_dst_agg == 'poolmean':
            self.pooling_fun = pygnn.pool.global_mean_pool
        
        ## The head configuration
        head_input_dim = hidden_dim * 2 if self.src_dst_agg == 'concat'  and self.task_level == 'edge' else hidden_dim

        if self.task == 'regression':
            dim_out = 1
        elif self.task =='classification':
            dim_out = args.num_classes
        else:
            raise ValueError('Invalid task')
        
        # head MLP layers
        self.head_layers = MLP(
            in_channels=head_input_dim, 
            hidden_channels=hidden_dim, 
            out_channels=dim_out, 
            num_layers=args.num_head_layers, 
            use_bn=False, dropout=0.0, 
            activation=args.act_fn,
        )

        ## Batch normalization
        self.use_bn = args.use_bn
        self.bn_node_x = nn.BatchNorm1d(hidden_dim)
        if self.use_bn and self.use_cl:
            print("[Warning] Using batch normalization with contrastive learning may cause performance degradation.")

        ## activation setting
        if args.act_fn == 'relu':
            self.activation = nn.ReLU()
        elif args.act_fn == 'elu':
            self.activation = nn.ELU()
        elif args.act_fn == 'tanh':
            self.activation = nn.Tanh()
        elif args.act_fn == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif args.act_fn == 'prelu':
            self.activation = nn.PReLU()
        else:
            raise ValueError('Invalid activation')
        
        ## Dropout setting
        self.drop_out = args.dropout
    

    def forward(self, batch):
        ## Node type / Edge type encoding
        x = self.node_encoder(batch.node_type)
        xe = self.edge_encoder(batch.edge_type)

        ## Contrastive learning encoder
        if self.use_cl:
            xcl = self.cl_linear(batch.x)
            ## concatenate node embeddings and embeddings learned by SGRL
            x = torch.cat((x, xcl), dim=1)

        
        ## If we use circuit statistics encoder
        if self.use_stats:
            net_node_mask = batch.node_type == NET
            dev_node_mask = batch.node_type == DEV
            pin_node_mask = batch.node_type == PIN
            ## circuit statistics embeddings (C in EQ.6)
            node_attr_emb = torch.zeros(
                (batch.num_nodes, self.c_embed_dim), device=batch.x.device
            )
            node_attr_emb[net_node_mask] = \
                self.net_attr_layers(batch.node_attr[net_node_mask])
            node_attr_emb[dev_node_mask] = \
                self.dev_attr_layers(batch.node_attr[dev_node_mask])
            node_attr_emb[pin_node_mask] = \
                self.pin_attr_layers(batch.node_attr[pin_node_mask, 0].long())
            ## concatenate node embeddings and circuit statistics embeddings (C in EQ.6)
            x = torch.cat((x, node_attr_emb), dim=1)

        # GNN layers
        for conv in self.layers:
            ## for models that also take edge_attr as input
            if self.model == 'gine' or self.model == 'resgatedgcn':
                x = conv(x, batch.edge_index, edge_attr=xe)
            else:
                x = conv(x, batch.edge_index)

            if self.use_bn:
                x = self.bn_node_x(x)
            
            x = self.activation(x)

            if self.drop_out > 0.0:
                x = F.dropout(x, p=self.drop_out, training=self.training)

        ## task level : node
        if self.task_level == 'node':
            if self.net_only:
                net_node_mask = batch.node_type == NET
                pred = self.head_layers(x[net_node_mask])
                true_class = batch.y[:, 1][net_node_mask].long()
                true_label = batch.y[net_node_mask]
            else:
                pred = self.head_layers(x)
                true_class = batch.y[:, 1].long()
                true_label = batch.y

        elif self.task_level == 'edge':
            if self.src_dst_agg[:4] == 'pool':
                graph_emb = self.pooling_fun(x, batch.batch)
            ## Otherwise, only 2 embeddings from the anchor nodes are used to final prediction.
            else:
                batch_size = batch.edge_label.size(0)
                ## In the LinkNeighbor loader, the first batch_size nodes in x are source nodes and,
                ## the second 'batch_size' nodes in x are destination nodes. 
                ## Remaining nodes are their '1-hop', '2-hop', 'n-hop' neighbors.
                src_emb = x[:batch_size, :]
                dst_emb = x[batch_size:batch_size*2, :]
                if self.src_dst_agg == 'concat':
                    graph_emb = torch.cat((src_emb, dst_emb), dim=1)
                else:
                    graph_emb = src_emb + dst_emb

            pred = self.head_layers(graph_emb)
            true_class = batch.edge_label[:, 1].long()
            true_label = batch.edge_label
        
        else:
            raise ValueError('Invalid task level')
            
        return pred,true_class,true_label
        
        
    

