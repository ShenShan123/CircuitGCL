from sklearn.multiclass import OneVsRestClassifier
from sgrl_models import CustomConv, CustomOnline, Target
import torch
import time
from tqdm import tqdm
import sys
import os

from sram_dataset import adaption_for_sgrl
from torch_geometric.loader import NeighborLoader, ShaDowKHopSampler, LinkNeighborLoader

def train_online_encoder(online: CustomOnline, optimizer, loader, graph_adj, device):
    tot_loss = 0.0
    num_loss = 0
    online.train()

    for batch in tqdm(iterable=loader, desc='Online batches'):
        optimizer.zero_grad()
        batch = batch.to(device)
        h,h_pred,h_target = online(batch)
        loss = online.get_loss(h_pred, h_target.detach())
        loss.backward()
        tot_loss += loss.item()
        num_loss += 1
        optimizer.step()

    online.update_target_encoder()
    return tot_loss / num_loss

def train_target_encoder(target: Target, optimizer, loader, device):
    tot_loss = 0.0
    num_loss = 0
    target.train()

    for batch in tqdm(iterable=loader, desc='Target batches', leave=False):
        optimizer.zero_grad()
        batch = batch.to(device)
        h_target = target(batch)
        loss = target.get_loss(h_target)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        num_loss += 1

    return tot_loss / num_loss

def adj_norm(data):
    nb_nodes = data.x.size(0)
    self_loop_for_adj = torch.Tensor([i for i in range(nb_nodes)]).unsqueeze(0)
    self_loop_for_adj = torch.concat([self_loop_for_adj, self_loop_for_adj], dim=0)
    
    slsp_adj = torch.concat([data.edge_index.cpu(), self_loop_for_adj], dim=1)
    slsp_adj = torch.sparse_coo_tensor(slsp_adj.long(), torch.ones(slsp_adj.size()[1]),
                                        torch.Size([nb_nodes, nb_nodes]))
    adj_t = slsp_adj
    deg = torch.sparse.sum(adj_t, dim=1).to_dense()
    deg_inv_sqrt = deg.pow(-0.5)
    adj_t = adj_t * deg_inv_sqrt.view(1, -1)
    adj_t = adj_t * deg_inv_sqrt.view(-1, 1)
    return adj_t

def get_all_contrastive_embed(
        online_model, pkl_path, 
        train_graph, loader, 
        hidden_dim, num_hop, device
    ):
    """
    Get all node embeddings from the online encoder of SGRL model.
    Args:
        online_model (CustomOnline): The online encoder of SGRL model
        pkl_path (str): The path to the state_dict file of the online encoder
        train_graph (torch_geometric.data.Batch): The training graph
        loader (torch_geometric.loader.NeighborLoader): The loader for the dataset
        hidden_dim (int): The hidden dimension of the model
        num_hop (int): The number of hops
        device (torch.device): The device
    Returns:
        torch.Tensor: The embeddings of all nodes in the graph
    """
    print(f"Loading model from {pkl_path}")
    online_model.load_state_dict(torch.load(pkl_path))
    online_model.eval()

    # Initialize all CL embeddings
    embeds = torch.zeros((train_graph.num_nodes, hidden_dim), requires_grad=False)
    
    for batch in tqdm(iterable=loader, desc='Getting CL embeddings', leave=False):
        # slsp_adj = graph_adj.index_select(0, batch.n_id)
        # batch.slsp_adj = slsp_adj.index_select(1, batch.n_id)
        # assert batch.input_id.size(0) == batch.batch_size, \
        #   f"input_id size {batch.input_id.size(0)} != batch_size {batch.batch_size}"
        batch = batch.to(device)

        ## We only record the embeddings of the sampled (specified by batch.input_id) 
        ## nodes in this batch, even though the model return all neighbors' embeddings in the batch.
        ## The first batch_size embeddings are the embeddings of the sampled nodes. 
        ## See docs about pyg loader.
        embeds[batch.input_id] = \
            online_model.embed(batch, num_hop).detach().cpu()[:batch.input_id.size(0)]
    
    ## Assign the embeddings to the batched big training graph
    # train_graph.cl_embed = embeds
    online_model = online_model.cpu()

    # ## We slice node embeds in the large `train_graph` 
    # ## and map them back to the corresponding each dataset
    # cl_embeds_for_dataset = [
    #     embeds[ train_graph.ptr[i] : train_graph.ptr[i+1] ] 
    #     for i in range(train_graph.num_graphs)
    # ]

    # return cl_embeds_for_dataset
    return embeds

def sgrl_train(args, dataset, device):
    """
    Training SGRL model.
    Args:
        args (argparse.Namespace): The arguments for SGRL
        dataset (torch_geometric.data.InMemoryDataset): The dataset
        device (torch.device): The device
    Returns:
        torch.Tensor: The embeddings of all nodes in dataset
    """
    e1_lr = args.e1_lr
    e2_lr = args.e2_lr
    weight_decay = args.weight_decay
    hidden_dim = args.cl_hid_dim
    activation = args.cl_act_fn
    num_layers = args.cl_gnn_layers
    num_epochs = args.cl_epochs
    dropout = args.cl_dropout
    momentum = args.momentum
    train_graph = adaption_for_sgrl(dataset)
    train_adj = adj_norm(train_graph)

    #========== model construction ==========#
    num_hop = args.num_hops
    input_dim = train_graph.x.size(1)
    online_conv = CustomConv(
        args#.cl_model, input_dim,hidden_dim,hidden_dim,activation,num_layers, 
        #drop_out=dropout,
    ).to(device)
    target_conv = CustomConv(
        args#.cl_model, input_dim,hidden_dim,hidden_dim,activation,num_layers, 
        #drop_out=dropout,
    ).to(device)

    # online_model = Online(online_conv,target_conv,hidden_dim,slsp_adj,num_hop,momentum).to(device)
    online_model = CustomOnline(online_conv,target_conv,hidden_dim,num_hop,momentum).to(device)
    target_model = Target(target_conv).to(device)
    
    online_optimizer = torch.optim.Adam(online_model.parameters(), lr=e1_lr, weight_decay=weight_decay)
    target_optimizer = torch.optim.Adam(target_model.parameters(), lr=e2_lr, weight_decay=weight_decay)

    best_online_loss = 1e9
    best_target_loss = 1e9

    #========== contrastive learning ==========#
    batch_size=args.cl_batch_size #4096
    train_graph_loader = NeighborLoader(
        train_graph,
        num_neighbors=[args.cl_num_neighbors,] * num_layers,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        # directed=False,
    )
    # kwargs = {
    # 'batch_size': batch_size, 'shuffle': True, 'num_workers': 8, 
    # 'drop_last': True, 'pin_memory': True
    # }
    # train_graph_loader = ShaDowKHopSampler(
    # data=train_graph, depth=num_layers, 
    # num_neighbors=32, node_idx=None, **kwargs)

    tag = str(time.time())
    model_name = ""
    best_epoch = 0
    cnt_wait = 0

    model_name = f"pkl/pkl_online/best_online_{args.dataset}_" + \
        f"{args.cl_model}_layer{num_layers}_" + \
        f"dim{hidden_dim}_{activation}_dr{dropout:.1f}_small.pkl" # small g SGRL
  

    if not os.path.exists(model_name):
        print(f"Training SGRL model with name {model_name}...")

        for epoch in range(num_epochs):
            online_optimizer.zero_grad()
            target_optimizer.zero_grad()
            
            # online_loss = train_online(online_model,online_optimizer,data)  
            # online_loss = train_customonline(online_model, online_optimizer, train_graph)        
            online_loss = train_online_encoder(
                online_model, online_optimizer, 
                train_graph_loader, train_adj,  device)        
            if online_loss < best_online_loss:
                best_online_loss = online_loss
                best_epoch = epoch
                torch.save(online_model.state_dict(), model_name)
                cnt_wait = 0
            
            # target_loss = train_target(target_model,target_optimizer,train_graph)
            target_loss = train_target_encoder(
                target_model, target_optimizer, train_graph_loader, device)
            
            if target_loss < best_target_loss:
                best_target_loss = target_loss
                torch.save(target_model.state_dict(), 
                           model_name.replace('online', 'target'))

            print(f"Epoch:{epoch} online_loss={online_loss:.6f} target_loss={target_loss:.6f}")

            if (online_loss < -0.99 and target_loss < -0.99) or cnt_wait == 20:
                print("Do early stop")
                break
            else:
                cnt_wait += 1

    #========== get all node embeddings learnt by SGRL ==========#
    return get_all_contrastive_embed(
        online_model, model_name, train_graph, 
        train_graph_loader, hidden_dim, num_hop, device
    )

# if __name__ == '__main__':
#     # warnings.filterwarnings("ignore")
#     parser = argparse.ArgumentParser('SGRL')
#     parser.add_argument('--dataset_name', type=str, default='Photo', help='dataset_name')
#     parser.add_argument('--data_dir', type=str, default='../../datasets', help='data_dir')
#     parser.add_argument('--log_dir', type=str, default='./log/log_Photo', help='log_dir')
#     parser.add_argument('--e1_lr', type=float, default=0.001, help='online_learning_rate')
#     parser.add_argument('--e2_lr', type=float, default=0.001, help='target_learning_rate')
#     parser.add_argument('--momentum', type=float, default=0.99, help='EMA')
#     parser.add_argument('--weight_decay', type=float, default=0., help='weight_decay')
#     parser.add_argument('--num_epochs', type=int, default=700, help='num_epochs')
#     parser.add_argument('--seed', type=int, default=66666, help='seed')
#     parser.add_argument('--hidden_dim', type=int, default=1024, help='hidden_dim')
#     parser.add_argument('--num_layers', type=int, default=1, help='num_layers')
#     parser.add_argument('--num_hop', type=int, default=1, help='num_hop')
#     parser.add_argument('--trials', type=int, default=20, help='trials')
#     args = parser.parse_args()  
#     contrastive_train(args)