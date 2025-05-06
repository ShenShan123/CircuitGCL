import argparse
import torch
import numpy as np
from sram_dataset import performat_SramDataset, adaption_for_sgrl
from downstream_train import downstream_train
import os
import random
from sgrl_train import sgrl_train
import datetime
import sys

if __name__ == "__main__":
    # STEP 0: Parse Arguments ======================================================================= #
    parser = argparse.ArgumentParser(description="CircuitGCL")
    # Task setting
    parser.add_argument("--task_level", type=str, default="edge", help="Task level. 'node' or 'edge'.")
    parser.add_argument("--task", type=str, default="classification", help="Task type. 'classification' or 'regression'.")
    
    # Dataset setting
    parser.add_argument("--dataset", type=str, default="ssram+digtime+timing_ctrl+array_128_32_8t", help="Names of datasets.") # the first dataset is the training dataset
    parser.add_argument('--neg_edge_ratio',type=float,default=0.0,help='The ratio of negative edges.') # 0.0 for classification, 0.5 for regression
    parser.add_argument('--net_only',type=bool,default=True,help='Only use net nodes for node level task or not.')

    # Graph sampling setting
    parser.add_argument("--small_dataset_sample_rates", type=float, default=1.0, help="The sample rate for small dataset.")
    parser.add_argument("--large_dataset_sample_rates", type=float, default=0.1, 
                        help='Target edge num of large dataset. 20% for large G')
    parser.add_argument("--num_hops", type=int, default=4, help="Number of hops in subgraph sampling.")
    parser.add_argument('--num_neighbors',type=int,default=64,help='The number of neighbors in subgraph sampling.')
    
    # Training setting
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument("--num_workers", type=int, default=8, help="The number of workers in data loaders.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index. Default: -1, using cpu.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="The batch size.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")

    # SGRL arguments
    parser.add_argument('--sgrl', type=int, default=0, help='Enable contrastive learning, i.e., SGRL.')
    parser.add_argument('--e1_lr', type=float, default=1e-6, help='Learning rate for online encoder in SGRL.')
    parser.add_argument('--e2_lr', type=float, default=2e-7, help='Learning rate for target encoder in SGRL.')
    parser.add_argument('--momentum', type=float, default=0.99, help='EMA')
    parser.add_argument('--weight_decay', type=float, default=0., help='weight_decay')
    parser.add_argument('--cl_epochs', type=int, default=800, help='cl_epochs')
    parser.add_argument("--cl_model", type=str, default='clustergcn', 
                        choices=['clustergcn', 'resgatedgcn', 'gat', 'gcn', 'sage', 'gine'],
                        help="The gnn model of SGRL encoders.")
    parser.add_argument('--cl_act_fn', default='tanh', choices=['relu', 'elu', 'tanh', 'leakyrelu', 'prelu'], help='Activation function of SGRL encoders')
    parser.add_argument("--cl_gnn_layers", type=int, default=4, help="Number of GNN layers of encoders in SGRL.")
    parser.add_argument('--cl_hid_dim', type=int, default=256, help='hidden_dim for contrastive learning')
    parser.add_argument('--cl_batch_size', type=int, default=1024, 
                        help='Batch size for contrastive learning. 512 for large G')
    parser.add_argument('--cl_num_neighbors', type=int, default=64, 
                        help='Number of neighbors for contrastive learning. 128 for large G')
    parser.add_argument('--cl_dropout', type=float, default=0.3, help='Dropout for SGRL encoders.')
    
    ## Downstream GNN setting
    parser.add_argument("--model", type=str, default='sage', help="The gnn model. Could be 'clustergcn', 'resgatedgcn', 'gat', 'gcn', 'sage', 'gine'.")
    parser.add_argument("--num_gnn_layers", type=int, default=4, help="Number of GNN layers.")
    parser.add_argument("--num_head_layers", type=int, default=2, help="Number of head layers.")
    parser.add_argument("--hid_dim", type=int, default=144, help="Hidden layer dim.")
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout for neural networks.')
    parser.add_argument('--use_bn', type=int, default=0, help='0 or 1. Batch norm for neural networks.')
    parser.add_argument('--act_fn', default='prelu', choices=['relu', 'elu', 'tanh', 'leakyrelu', 'prelu'], help='Activation function')
    parser.add_argument('--use_stats', type=int, default=1, help='0 or 1. Circuit statistics features. Use in node task.')

    # Regression setting
    parser.add_argument('--src_dst_agg', type=str, default='concat', choices=['concat', 'add', 'pooladd', 'poolmean'],
                        help="The way to aggregate nodes. Can be 'concat' or 'add' or 'pooladd' or 'poolmean'.")
    parser.add_argument("--regress_loss", type=str, default='mse', choices=['mse', 'gai', 'bmc', 'bni', 'lds'], help="The loss function for edge regression. Could be 'mse', 'bmc', or 'gai'.")
    
    # Classification setting
    parser.add_argument('--class_loss',type=str,default='bsmCE',choices=['bsmCE','focal','cross_entropy'],help='The loss function for classification.')
    parser.add_argument('--num_classes',type=int,default=5,help='The number of classes for node classification.')
    parser.add_argument('--class_boundaries',type=list,default=[0.2,0.4,0.6,0.8],help='The boundaries for classification.')

    # Balanced MSE setting for GAI implementation
    parser.add_argument("--noise_sigma", type=float, default=0.001, help="The simga_noise of Balanced MSE (EQ 3.6).")
    
    # LDS setting
    parser.add_argument('--lds_kernel', type=str, default='gaussian',
                    choices=['gaussian', 'triang', 'laplace'], help='LDS kernel type')
    parser.add_argument('--lds_ks', type=int, default=9, help='LDS kernel size: should be odd number. 5 0r 9.')
    parser.add_argument('--lds_sigma', type=float, default=0.02, help='LDS gaussian/laplace kernel sigma. 1 or 2.')
    
    parser.add_argument('--log_dir', type=str, default='logs', help='The directory to save the log file.')

    args = parser.parse_args()

    # Syncronize all random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    ##set log file
    
    # create log file
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.task == 'classification':
        log_filename = os.path.join(args.log_dir, f"{timestamp}_{args.task_level}_{args.task}_{args.dataset}_loss{args.class_loss}_batch{args.batch_size}.txt")
    else: # regression task
        log_filename = os.path.join(args.log_dir, f"{timestamp}_{args.task_level}_{args.task}_{args.dataset}_loss{args.regress_loss}_batch{args.batch_size}.txt")
    log_file = open(log_filename, 'w')
    
    # Redirect standard output to both file and console
    class Tee(object):
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)


    # Check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.gpu))
        print('Using GPU: {}'.format(args.gpu))
    else:
        device = torch.device("cpu")

    print(f"============= PID = {os.getpid()} ============= ")
    print(args)


    # STEP 1: Load Dataset =================================================================== #
    dataset = performat_SramDataset(
        name=args.dataset, 
        dataset_dir='./datasets/', 
        neg_edge_ratio=args.neg_edge_ratio,
        to_undirected=True,
        small_dataset_sample_rates=args.small_dataset_sample_rates,
        large_dataset_sample_rates=args.large_dataset_sample_rates,
        task_level=args.task_level,
        net_only=args.net_only,
        class_boundaries=args.class_boundaries
    )

    # STEP 2-3: If you do graph contrastive learning, you should add the code here =========== #
    if args.sgrl == 1:
        embedding_path = f'./embeddings/'
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
        embedding_path = os.path.join(embedding_path, f'embeddings_{args.dataset}_layer{args.num_gnn_layers}_dim{args.hid_dim}_{args.act_fn}.pkl')
        if os.path.exists(embedding_path):
            cl_embeds = torch.load(embedding_path)
        else:
            cl_embeds = sgrl_train(args, dataset, device)
            torch.save(cl_embeds, embedding_path)
    else:
        cl_embeds = None

    # STEP 4: Training Epochs ================================================================ #

    downstream_train(args, dataset, device, cl_embeds)

    sys.stdout = original_stdout
    log_file.close()
    print(f"Finished running and save results to {log_filename}")