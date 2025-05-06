import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, ResGatedGraphConv, GINConv, ChebConv, GINEConv, ClusterGCNConv
from torch_geometric.utils import dropout_edge,mask_feature
from torch_geometric.nn.models.mlp import MLP

NET = 0
DEV = 1
PIN = 2
    
class CustomConv(torch.nn.Module):
    def __init__(self, args):
                 #model, input_dim,hidden_dim,proj_dim,activation,num_layers, drop_out=0.0):
        super(CustomConv,self).__init__()
        hidden_dim = args.cl_hid_dim
        ## activation setting
        if args.cl_act_fn == 'relu':
            self.activation = nn.ReLU()
        elif args.cl_act_fn == 'elu':
            self.activation = nn.ELU()
        elif args.cl_act_fn == 'tanh':
            self.activation = nn.Tanh()
        elif args.cl_act_fn == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif args.cl_act_fn == 'prelu':
            self.activation = nn.PReLU()
        else:
            raise ValueError('Invalid activation')
        
        self.use_stats = False

        ## Circuit Statistics encoder, producing matrix C
        if self.use_stats:
            print("Using circuit statistics in SGRL ...")
            node_embed_dim = hidden_dim // 2
            ## add node_attr transform layer for net/device/pin nodes, by shan
            self.net_attr_layers = nn.Linear(17, node_embed_dim, bias=True)
            self.dev_attr_layers = nn.Linear(17, node_embed_dim, bias=True)
            ## pin attributes are {0, 1, 2} for gate pin, source/drain pin, and base pin
            self.pin_attr_layers = nn.Embedding(17, node_embed_dim)
            self.c_embed_dim = node_embed_dim
        else:
            node_embed_dim = hidden_dim

        ## Node/Edge type encoders.
        self.node_type_embed = nn.Embedding(6, node_embed_dim)
        self.edge_type_embed = nn.Embedding(8, node_embed_dim)

        self.layers = nn.ModuleList()
        self.model = args.cl_model

        for _ in range(args.cl_gnn_layers):
            ## the following are examples of using different GNN layers
            if self.model == 'clustergcn':
                self.layers.append(ClusterGCNConv(hidden_dim, hidden_dim))
            elif self.model == 'gcn':
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
            elif self.model == 'sage':
                self.layers.append(SAGEConv(hidden_dim, hidden_dim))
            elif self.model == 'gat':
                self.layers.append(GATConv(hidden_dim, hidden_dim, heads=1))
            elif self.model == 'resgatedgcn':
                self.layers.append(ResGatedGraphConv(hidden_dim, hidden_dim, edge_dim=hidden_dim))
            elif self.model == 'gine':
                mlp = MLP(
                    in_channels=hidden_dim, 
                    hidden_channels=hidden_dim, 
                    out_channels=hidden_dim, 
                    num_layers=2, 
                    norm=None,
                    act=self.activation,
                )
                self.layers.append(GINEConv(mlp, train_eps=True, edge_dim=hidden_dim))
            else:
                raise ValueError(f'Unsupported GNN model: {self.model}')


        self.use_bn = args.use_bn
        self.bn_node_x = nn.BatchNorm1d(hidden_dim)
        self.drop_out = args.cl_dropout

        proj_dim = hidden_dim
        
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim,proj_dim),
            ## The PReLU has trainable parameters, so another instance is needed
            self.activation if args.cl_act_fn != 'prelu' else nn.PReLU(),
            torch.nn.Dropout(args.cl_dropout),
            torch.nn.Linear(proj_dim,proj_dim)
        )  
        
    def forward(self,batch):
        # print(batch)
        ## Node type / Edge type encoding
        z = self.node_type_embed(batch.x).squeeze()
        ze = self.edge_type_embed(batch.x).squeeze()

        ## If we use circuit statistics encoder
        if self.use_stats:
            net_node_mask = (batch.x == NET).squeeze()
            dev_node_mask = (batch.x == DEV).squeeze()
            pin_node_mask = (batch.x == PIN).squeeze()
            ## circuit statistics embeddings (C in EQ.6)
            node_attr_emb = torch.zeros(
                (batch.num_nodes, self.c_embed_dim), device=batch.x.device
            )
            # print('node_attr_emb.shape', node_attr_emb.shape)
            # print('batch.node_attr.shape', batch.node_attr.shape)
            # print('self.c_embed_dim', self.c_embed_dim)
            # print('batch.node_attr[net_node_mask]', batch.node_attr[net_node_mask])
            node_attr_emb[net_node_mask] = \
                self.net_attr_layers(batch.node_attr[net_node_mask])
            node_attr_emb[dev_node_mask] = \
                self.dev_attr_layers(batch.node_attr[dev_node_mask])
            node_attr_emb[pin_node_mask] = \
                self.pin_attr_layers(batch.node_attr[pin_node_mask, 0].long())
            ## concatenate node embeddings and circuit statistics embeddings (C in EQ.6)
            z = torch.cat((z, node_attr_emb), dim=1)

        for conv in self.layers:
            if self.model == 'gine' or self.model == 'resgatedgcn':
                z = conv(z, batch.edge_index, edge_attr=ze)
            else:
                z = conv(z, batch.edge_index)

            if self.use_bn:
                z = self.bn_node_x(z)

            z = self.activation(z)
            
            if self.drop_out > 0.0:
                z = F.dropout(z, p=self.drop_out, training=self.training)

        return z, self.projection_head(z)

class CustomOnline(torch.nn.Module):
    def __init__(self,online_encoder,target_encoder,hidden_dim,num_hop,momentum):
        super(CustomOnline,self).__init__()
        self.online_encoder = online_encoder
        self.target_encoder = target_encoder
        self.num_hop = num_hop
        self.momentum = momentum
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.PReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        
    def update_target_encoder(self):
        for p, new_p in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            next_p = self.momentum * p.data + (1 - self.momentum) * new_p.data
            p.data = next_p
            
    def forward(self, batch):
        h = self.embed(batch, self.num_hop)
        h_pred = self.predictor(h)
        with torch.no_grad():
               h_target,_ = self.target_encoder(batch)
              
        return h,h_pred,h_target
       
    def get_loss(self,z1,z2):
        z1 = F.normalize(z1, dim=-1, p=2)
        z2 = F.normalize(z2, dim=-1, p=2)
        loss = (z1 * z2).sum(dim=-1)
        loss = -loss.mean()
        return loss
    
    def embed(self, batch, Globalhop=10):
        h_1,_ = self.online_encoder(batch)
        return h_1
        #NOTE: The following code is not used in the current implementation
        #TODO: Compare the performance with/without slsp_adj
        h_2 = h_1.clone()
        for _ in range(Globalhop):
            h_2 = batch.slsp_adj @ h_2
        return h_1 + h_2
    
    
class Target(torch.nn.Module):
    def __init__(self,target_encoder):
        super(Target,self).__init__()
        self.target_encoder = target_encoder
        
    def forward(self,batch):
        h_target,_ = self.target_encoder(batch)
        return h_target
    
    def get_loss(self,z):
        z = F.normalize(z,dim=-1, p=2)
        return -(z - z.mean(dim=0)).pow(2).sum(1).mean()