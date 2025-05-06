import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score,
    mean_absolute_error, mean_squared_error,
    root_mean_squared_error, r2_score,
)
import numpy as np
import time
from tqdm import tqdm
from model import GraphHead
from sampling import dataset_sampling
from balanced_mse import GAILoss, BMCLoss, BNILoss, train_gmm, WeightedMSE, get_lds_weights, BalancedSoftmax, FocalLoss, compute_class_weights
import os
import matplotlib.pyplot as plt

# from torch.utils.data.sampler import SubsetRandomSampler
# from sram_dataset import LinkPredictionDataset
# from sram_dataset import collate_fn, adaption_for_sgrl
# from torch_geometric.data import Batch

# from torch_geometric.loader import NeighborLoader, GraphSAINTRandomWalkSampler, GraphSAINTEdgeSampler, ShaDowKHopSampler

NET = 0
DEV = 1
PIN = 2

class Logger (object):
    """ 
    Logger for printing message during training and evaluation. 
    Adapted from GraphGPS 
    """
    
    def __init__(self, task='classification', max_label=None):
        super().__init__()
        # Whether to run comparison tests of alternative score implementations.
        self.test_scores = False
        self._iter = 0
        self._true = []
        self._pred = []
        self._loss = 0.0
        self._size_current = 0
        self.task = task
        self.max_label = max_label
    def update_stats(self, true, pred, batch_size, loss):
        self._true.append(true)
        self._pred.append(pred)
        self._size_current += batch_size
        self._loss += loss * batch_size
        self._iter += 1

    def write_epoch(self, split=""):
        true, pred_score = torch.cat(self._true), torch.cat(self._pred)
        true = true.numpy()
        pred_score = pred_score.numpy()
        reformat = lambda x: round(float(x), 4)        

        if self.task == 'classification':

            max_label_indices = np.where(true == self.max_label)[0]
            mask = np.zeros_like(true, dtype=bool)
            if len(max_label_indices) > 0:
                selected_max_indices = np.random.choice(
                    max_label_indices, 
                    size=max(1, len(max_label_indices) // 10),  # 至少保留1个样本
                    replace=False
                )
                mask[selected_max_indices] = True
                    
            mask[true != self.max_label] = True

            accuracy = accuracy_score(true[mask], pred_score[mask])
            f1 = f1_score(true[mask], pred_score[mask], average='macro')
            precision = precision_score(true[mask], pred_score[mask], average='macro')
            recall = recall_score(true[mask], pred_score[mask], average='macro')

            res = {
                'loss': reformat(self._loss / self._size_current),
                'accuracy': reformat(accuracy),
                'f1': reformat(f1),
                'precision': reformat(precision),
                'recall': reformat(recall),
            }

        else:  # regression task
            res = {
                'loss': round(self._loss / self._size_current, 8),
                'mae': reformat(mean_absolute_error(true, pred_score)),
                'mse': reformat(mean_squared_error(true, pred_score)),
                'rmse': reformat(root_mean_squared_error(true, pred_score)),
                'r2': reformat(r2_score(true, pred_score)),
            }

        # Just print the results to screen
        print(split, res)
        return res

def compute_loss(args, pred, true, criterion):
    """Compute loss and prediction score. 
    Args:
        args (argparse.Namespace): The arguments
        pred (torch.tensor): Unnormalized prediction
        true (torch.tensor): Groud truth label
        criterion (torch.nn.Module): The loss function
    Returns: Loss, normalized prediction score
    """
    assert criterion, "Loss function is not provided!"
    assert pred.size(0) == true.size(0), \
        "Prediction and true label size mismatch!"

    if args.task == 'classification':
        if args.class_loss == 'focal':
            class_weights = compute_class_weights(true, args.num_classes) 
            focal_loss = FocalLoss(gamma=2.0, class_weights=class_weights)
            loss = focal_loss(pred, true)
        else:
            loss = F.cross_entropy(pred, true)
        predict_class = torch.argmax(pred, dim=1)
        return loss, predict_class, true
      
    elif args.task == 'regression':
        ## Size of `pred` must be [N, 1] for regression task
        assert pred.ndim == 1 or pred.size(1) == 1
        pred = pred.view(-1, 1)

        assert (true.size(1) == 2), \
            "true label has two columns [continuous label, discrete label or label weights]!"
        
        ## for LDS loss, the second column of `true` is the weights
        if args.regress_loss == 'lds':
            loss = criterion(
                pred, 
                true[:, 0].squeeze(), 
                true[:, 1].squeeze() # the weight for each label
            )
            return loss, pred, true[:, 0].view(pred.size())

        ## for other loss func, the second column of true is the discrete label,
        ## which is not in use.
        ## Size of `true[:, 0]` is [N,] for regression task, 
        ## ensuring same sizes of `pred` and `true`
        true = true[:, 0] if true.ndim == 2 else true
        true = true.view(-1, 1)
        
        return criterion(pred, true), pred, true
    
    else:
        raise ValueError(f"Task type {args.task} not supported!")

@torch.no_grad()
def eval_epoch(args, loader, model, device, 
               split='val', criterion=None):
    """ 
    evaluate the model on the validation or test set
    Args:
        args (argparse.Namespace): The arguments
        loader (torch.utils.data.DataLoader): The data loader
        model (torch.nn.Module): The model
        device (torch.device): The device to run the model on
        split (str): The split name, 'val' or 'test'
        criterion (torch.nn.Module): The loss function
    """
    model.eval()
    time_start = time.time()
    logger = Logger(task=args.task)

    for i, batch in enumerate(tqdm(loader, desc="eval_"+split, leave=False)):
        pred, class_true, label_true = model(batch.to(device))
        if args.task == 'regression':
            loss, pred_score, true = compute_loss(args, pred, label_true, criterion=criterion)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)
            logger.update_stats(true=_true,
                                pred=_pred,
                                batch_size=_true.size(0),
                                loss=loss.detach().cpu().item(),
                                )
        elif args.task == 'classification':
            loss, predict_class, true = compute_loss(args, pred, class_true, criterion=criterion)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = predict_class.detach().to('cpu', non_blocking=True)
            logger.update_stats(true=_true,
                                pred=_pred,
                                batch_size=_true.size(0),
                                loss=loss.detach().cpu().item(),
                                )
    return logger.write_epoch(split)

def regress_train(args, regressor, optimizier, criterion,
          train_loader, val_loader, test_loaders, max_label,
          device):
    """
    Train the head model for regression task
    Args:
        args (argparse.Namespace): The arguments
        regressor (torch.nn.Module): The regressor
        optimizier (torch.optim.Optimizer): The optimizer
        criterion (torch.nn.Module): The loss function
        train_loader (torch.utils.data.DataLoader): The training data loader
        val_loader (torch.utils.data.DataLoader): The validation data loader  
        test_laders (list): A list of test data loaders
        device (torch.device): The device to train the model on
    """
    optimizier.zero_grad()
    
    best_results = {
        'best_val_mse': 1e9, 'best_val_loss': 1e9, 
        'best_epoch': 0, 'test_results': []
    }
    
    for epoch in range(args.epochs):
        logger = Logger(task=args.task, max_label=max_label)
        regressor.train()

        for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch:{epoch}')):
            optimizier.zero_grad()

            ## Get the prediction from the model
            y_pred,y_class, y = regressor(batch.to(device))
            loss, pred, true = compute_loss(args, y_pred, y, criterion=criterion)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = y_pred.detach().to('cpu', non_blocking=True)

            loss.backward()
            optimizier.step()
            
            ## Update the logger and print message to the screen
            logger.update_stats(
                true=_true, pred=_pred, 
                batch_size=_true.squeeze().size(0), 
                loss=loss.detach().cpu().item()
            )

        logger.write_epoch(split='train')
        ## ========== validation ========== ##
        val_res = eval_epoch(
            args, val_loader, 
            regressor, device, split='val', criterion=criterion
        )

        ## update the best results so far
        if best_results['best_val_mse'] > val_res['mse']:
            best_results['best_val_mse'] = val_res['mse']
            best_results['best_val_loss'] = val_res['loss']
            best_results['best_epoch'] = epoch
        
            test_results = []
           
            ## ========== testing on other datasets ========== ##
            for test_name in test_loaders.keys():
                res = eval_epoch(
                    args, test_loaders[test_name], 
                    regressor, device, split='test', 
                    criterion=criterion
                )
                test_results.append(res)
            os.makedirs("downstream_model", exist_ok=True)
            torch.save(regressor.state_dict(), f"downstream_model/model_{epoch}-{args.regress_loss}.pth")

        if best_results['best_epoch'] == epoch:
            best_results['test_results'] = test_results

        print( "=====================================")
        print(f" Best epoch: {best_results['best_epoch']}, mse: {best_results['best_val_mse']}, loss: {best_results['best_val_loss']}")
        print(f" Test results: {[res for res in best_results['test_results']]}")
        print( "=====================================")

def class_train(args, classifier,optimizer_classifier, 
          train_loader, val_loader, test_loaders, max_label,
          device):
    """
    Train model for capacitance classification task
    Args:
        args (argparse.Namespace): The arguments
        classifier (torch.nn.Module): The classifier
        optimizer_classifier (torch.optim.Optimizer): The optimizer for the classifier
        train_loader (torch.utils.data.DataLoader): The training data loader
        val_loader (torch.utils.data.DataLoader): The validation data loader  
        test_laders (list): A list of test data loaders
        device (torch.device): The device to train the model on
    """
    # Reset optimizers
    optimizer_classifier.zero_grad()

    # create the directory to save the model
    classifier_save_dir = os.path.join("models_node_cap_classifier")
    
    os.makedirs(classifier_save_dir, exist_ok=True)
    
    # initialize the best model metrics

    best_f1 = 0.0
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        # add the logger for classification task
        logger = Logger(task='classification', max_label=max_label)
        classifier.train()

        for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch:{epoch}')):
            # Move batch to device
            batch = batch.to(device)
            optimizer_classifier.zero_grad()
            
            ## Get the prediction from the model
            class_logits,true_class, true_label = classifier(batch)
            class_probs = F.softmax(class_logits, dim=1)

            predict_class = torch.argmax(class_probs, dim=1)
            
            ## set the loss function for classification task
            if args.class_loss == 'focal':
                # calculate the class weights
                num_classes = class_logits.size(1)
                class_weights = compute_class_weights(true_class, num_classes)
                # apply Focal Loss
                criterion = FocalLoss(gamma=2.0, class_weights=class_weights)
                class_loss = criterion(class_logits, true_class)
            elif args.class_loss == 'bsmCE':
                ## calculate the number of samples per class
                sample_per_class = []
                for i in range(args.num_classes):
                    sample_per_class.append(torch.sum(true_class == i).item())
                # apply Balanced Softmax CE
                criterion = BalancedSoftmax(sample_per_class)
                class_loss = criterion(class_logits, true_class)

            elif args.class_loss == 'cross_entropy':
                criterion = torch.nn.CrossEntropyLoss()
                class_loss = criterion(class_logits, true_class)
            else:
                raise ValueError(f"Loss function {args.class_loss} not supported!")
                
            class_loss.backward()
            optimizer_classifier.step()
            
            # update the statistics of classification results
            _class_pred = predict_class.detach().to('cpu', non_blocking=True)
            _class_true = true_class.detach().to('cpu', non_blocking=True)
            logger.update_stats(
                true=_class_true, 
                pred=_class_pred, 
                batch_size=_class_true.squeeze().size(0), 
                loss=class_loss if isinstance(class_loss, float) else class_loss.detach().cpu().item()
            )

        print(f"\n===== Epoch {epoch}/{args.epochs} - Elapsed: {time.time() - epoch_start_time:.2f}s =====")
        print("Classification results:")
        logger.write_epoch(split='train')
        
        ## ========== validation ========== ##
        val_class_res = eval_epoch(
            args, val_loader,
            classifier, device, split='val', criterion=criterion
        )
        #visualize_tsne(classifier, val_loader, device, num_samples=2000)

        ## ========== testing on other datasets ========== ##
        test_class_results = {}           
        eval_flag = False

        # if the f1 of the current classifier model is the highest, save the best model
        if val_class_res['f1'] > best_f1:
            best_f1 = val_class_res['f1']
            eval_flag = True
        
        if eval_flag :
            for test_name in test_loaders.keys():
                print(test_name)
                test_class_res = eval_epoch(
                    args, test_loaders[test_name], 
                    classifier, device, split='test', criterion=criterion
                )
                test_class_results[test_name] = test_class_res
        
        print( "=====================================")
        print(f" Best epoch: {epoch}, f1: {val_class_res['f1']}")
        print(f" Test results: {[res for res in test_class_results.values()]}")
        print( "=====================================")


def downstream_train(args, dataset, device, cl_embeds=None):
    """ downstream task training for link prediction
    Args:
        args (argparse.Namespace): The arguments
        dataset (torch_geometric.data.InMemoryDataset): The dataset
        batch_index (torch.tensor): The batch index for all_node_embeds
        all_node_embeds (torch.tensor): The node embeddings come from the contrastive learning
        device (torch.device): The device to train the model on
    """
    if args.sgrl:
        dataset.set_cl_embeds(cl_embeds)

    dataset.norm_nfeat([NET, DEV])

    

    
    
    # Subgraph sampling for each dataset graph & PE calculation
    (
        train_loader, val_loader, test_loaders, max_label
    ) = dataset_sampling(args, dataset)


    if args.task == 'regression':
        # set the loss function for regression
        if args.regress_loss == 'gai':
            gmm_path = train_gmm(dataset)
            criterion = GAILoss(init_noise_sigma=args.noise_sigma, gmm=gmm_path, device=device)
        elif args.regress_loss == 'bmc':
            criterion = BMCLoss(init_noise_sigma=args.noise_sigma, device=device)
        elif args.regress_loss == 'bni':
            _, bin_edges, bin_count = get_lds_weights(
                dataset._data.edge_label[:, 1], 
                args.lds_kernel, args.lds_ks, args.lds_sigma
            )
            criterion = BNILoss(args.noise_sigma, bin_edges, bin_count,  device=device)
        elif args.regress_loss == 'mse':
            criterion = torch.nn.MSELoss(reduction='mean')
        elif args.regress_loss == 'lds':
            weights, _, _ = get_lds_weights(
                dataset._data.edge_label[:, 1], 
                args.lds_kernel, args.lds_ks, args.lds_sigma
            )
            dataset._data.edge_label[:, 1] = weights
            criterion = WeightedMSE()
        else:
            raise ValueError(f"Loss func {args.regress_loss} not supported!")
        
        start = time.time()
        model = GraphHead(args)
        model = model.to(device)
        optimizier = torch.optim.Adam(model.parameters(),lr=args.lr)
        
        regress_train(args, model, optimizier, criterion,
              train_loader, val_loader, test_loaders, max_label,
              device)
        
    elif args.task == 'classification':
        model = GraphHead(args)
        start = time.time()
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        class_train(args, model, optimizer, train_loader, val_loader, test_loaders, max_label,
              device)
    
    else:
        raise ValueError(f"Task type {args.task} not supported!")
        

    elapsed = time.time() - start
    timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed))
    print(f"Done! Training took {timestr}")
   






    
    
    

