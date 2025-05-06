import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import joblib
from sklearn.mixture import GaussianMixture
import time
from tqdm import tqdm

import torch
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d

def kl_divergence(p: torch.Tensor, q: torch.Tensor, bins: int = 10, 
                  epsilon: float = 1e-8, save_path: str = "logs/histograms.png") -> float:
    """
    Compute KL divergence and plot histograms of the distributions.
    
    Args:
        p (torch.Tensor): True distribution [N, 1]
        q (torch.Tensor): Pred distribution [N, 1]
        bins (int): Number of histogram bins
        epsilon (float): Smoothing factor
        save_path (str): Path to save histogram plot
        
    Returns:
        float: KL divergence value
    """
    # Convert to 1D numpy arrays for plotting
    p_np = p.flatten().cpu().numpy()
    q_np = q.flatten().cpu().numpy()
    
    # Create figure with subplots
    plt.figure(figsize=(12, 6))
    
    # Plot first distribution
    plt.subplot(1, 2, 1)
    plt.hist(p_np, bins=bins, alpha=0.5, density=True, label='Distribution P')
    plt.title("Distribution P")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    
    # Plot second distribution
    plt.subplot(1, 2, 2)
    plt.hist(q_np, bins=bins, alpha=0.5, density=True, label='Distribution Q', color='orange')
    plt.title("Distribution Q")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    
    # Save and close plot
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

    # Compute KL divergence (original calculation)
    min_val = min(p.min(), q.min()).item()
    max_val = max(p.max(), q.max()).item()
    
    p_counts = torch.histc(p, bins=bins, min=min_val, max=max_val) + epsilon
    q_counts = torch.histc(q, bins=bins, min=min_val, max=max_val) + epsilon
    
    p_probs = p_counts / p_counts.sum()
    q_probs = q_counts / q_counts.sum()
    
    kl = (p_probs * (torch.log(p_probs) - torch.log(q_probs))).sum().item()
    
    return kl

def train_gmm(dataset):
    start = time.time()
    graph_idx = 0
    train_labels = dataset[graph_idx].edge_label

    for i in range(graph_idx+1, len(dataset.names)):
        test_labels = dataset[i].edge_label
        # Compute KL divergence and save histograms
        kl_value = kl_divergence(
            test_labels, train_labels, bins=20, 
            save_path=f"logs/{dataset.names[i]}_distribution_comparison.png"
        )
        print(f"KL Divergence for {dataset.names[i]}: {kl_value:.4f}")

    print('Training labels curated')
    print('Fitting GMM...')
    gmm = GaussianMixture(n_components=8, random_state=0, verbose=2).fit(
        train_labels.reshape(-1, 1).cpu().numpy())
    
    gmm_dict = {}
    gmm_dict['means'] = gmm.means_
    gmm_dict['weights'] = gmm.weights_
    gmm_dict['variances'] = gmm.covariances_
    gmm_path = 'pkl/gmm/gmm.pkl'

    joblib.dump(gmm_dict, gmm_path)

    elapsed = time.time() - start
    timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed))
    print(f"Train gmm took {timestr}")
    return gmm_path


class GAILoss(_Loss):
    def __init__(self, init_noise_sigma, gmm, device):
        super(GAILoss, self).__init__()
        self.gmm = joblib.load(gmm)
        self.gmm = {k: torch.tensor(self.gmm[k]).to(device) for k in self.gmm}
        self.noise_sigma = torch.nn.Parameter(torch.tensor(float(init_noise_sigma), device=device))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        loss = gai_loss(pred, target, self.gmm, noise_var)
        return loss


def gai_loss(pred, target, gmm, noise_var):
    gmm = {k: gmm[k].reshape(1, -1).expand(pred.shape[0], -1) for k in gmm}
    # print('gmm', gmm)
    # print('pred', pred.shape)
    # assert 0
    mse_term = F.mse_loss(pred, target, reduction='none') / 2 / noise_var + 0.5 * noise_var.log()
    # print('mse_term', mse_term)
    sum_var = gmm['variances'] + noise_var
    balancing_term = - 0.5 * sum_var.log() - 0.5 * (pred.view(-1, 1) - gmm['means']).pow(2) / sum_var + gmm['weights'].log()
    balancing_term = torch.logsumexp(balancing_term, dim=-1, keepdim=True)
    loss = mse_term + balancing_term
    loss = loss * (2 * noise_var).detach()

    return loss.mean()


class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma, device):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(float(init_noise_sigma), device=device))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        loss = bmc_loss(pred, target, noise_var)
        return loss


def bmc_loss(pred, target, noise_var):
    logits = - 0.5 * (pred - target).pow(2) / noise_var
    loss = F.cross_entropy(logits.view(-1), torch.arange(pred.shape[0], dtype=torch.float32, device=pred.device))
    loss = loss * (2 * noise_var).detach()

    return loss


class BNILoss(_Loss):
    def __init__(self, init_noise_sigma, bucket_centers, bucket_weights, device):
        super(BNILoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma, device=device))
        self.bucket_centers = bucket_centers.clone().detach().requires_grad_(False).to(device)
        self.bucket_weights = bucket_weights.clone().detach().requires_grad_(False).to(device)

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        loss = bni_loss(pred, target, noise_var, self.bucket_centers, self.bucket_weights)
        return loss


def bni_loss(pred, target, noise_var, bucket_centers, bucket_weights):
    mse_term = F.mse_loss(pred, target, reduction='none') / 2 / noise_var

    num_bucket = bucket_centers.shape[0]
    bucket_center = bucket_centers.unsqueeze(0).repeat(pred.shape[0], 1)
    bucket_weights = bucket_weights.unsqueeze(0).repeat(pred.shape[0], 1)

    balancing_term = - 0.5 * (pred.expand(-1, num_bucket) - bucket_center).pow(2) / noise_var + bucket_weights.log()
    balancing_term = torch.logsumexp(balancing_term, dim=-1, keepdim=True)
    loss = mse_term + balancing_term
    loss = loss * (2 * noise_var).detach()
    return loss.mean()

class WeightedMSE(_Loss):
    def __init__(self):
        super(WeightedMSE, self).__init__()

    def forward(self, inputs, targets, weights):
        loss = (inputs - targets) ** 2
        loss *= weights
        loss = torch.mean(loss)
        return loss

def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window

def get_lds_weights(discrete_labels: torch.Tensor, lds_kernel: str, lds_ks: int, lds_sigma: float):
    """ Calculate the weights for LDS loss based on the discrete labels.
    Args:
        discrete_labels (torch.Tensor): Discrete labels of the data.
        lds_kernel (str): The kernel type for LDS. Options are 'gaussian', 'triang', 'laplace'.
        lds_ks (int): The kernel size. Should be odd.
        lds_sigma (float): The sigma value for the kernel.
    Returns:
        torch.Tensor: The scaled weights.
        torch.Tensor: The empirical bin edges.
        torch.Tensor: The empirical label distribution.
    """
    discrete_labels = discrete_labels.detach().cpu().long()

    # Calculate empirical (original) label distribution: [Nb,]
    emp_bins, emp_label_dist = discrete_labels.unique(return_counts=True)
    

    # lds_kernel_window: [ks,], here for example, we use gaussian, ks=5, sigma=2
    lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
    print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')

    # Calcualte smoothed label distribution: [Nb,]
    smoothed_value = convolve1d(
        emp_label_dist.numpy(), weights=lds_kernel_window, mode='constant')
    
    # Use re-weighting based on effective label distribution, sample-wise weights: [Ns,]
    eff_num_per_label = [smoothed_value[bin_index] for bin_index in discrete_labels]
    weights = [np.float32(1 / x) for x in eff_num_per_label]

    # Scaling weights
    scaling = len(weights) / np.sum(weights)
    weights = [scaling * x for x in weights]
    return torch.tensor(weights), emp_bins / emp_bins.max(), emp_label_dist / emp_label_dist.sum()

class BalancedSoftmax(_Loss):
    """
    Balanced Softmax Loss
    """
    def __init__(self, sample_per_class):

        super(BalancedSoftmax, self).__init__()
       
        self.sample_per_class = sample_per_class
    


    def forward(self, input, label,  reduction='mean'):
        return balanced_softmax_loss(logits=input, labels=label, sample_per_class=self.sample_per_class, reduction=reduction)
    

def balanced_softmax_loss(logits,labels, sample_per_class, reduction):
    """Compute the Balanced Softmax Cross Entropy Loss between `logits` and the ground truth `labels`.
    Args:
      logits: A float tensor of size [batch, no_of_classes].
      labels: A int tensor of size [batch].
      sample_per_class: A int tensor or list of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    if isinstance(sample_per_class, list):
        sample_per_class = torch.tensor(sample_per_class).type_as(logits)
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss

class FocalLoss(torch.nn.Module):
    """
    Focal Loss for Multi-Class Classification
    Args:
        gamma (float): focal parameter, control the weight of easy and hard samples, default 2.0
        reduction (str): loss reduction, 'mean', 'sum'  or 'none'
        class_weights (Tensor): weights for each class
    """
    def __init__(self, gamma=2.0, reduction='mean', class_weights=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights
        
    def forward(self, input, target):
        """
        Args:
            input: model output logits, shape [N, C]
            target: target class index, shape [N]
        """
        log_probs = F.log_softmax(input, dim=1)
        probs = torch.exp(log_probs)
        
        target_probs = probs.gather(1, target.unsqueeze(1))
        
        focal_weight = (1 - target_probs) ** self.gamma
        
        if self.class_weights is not None:
            alpha = self.class_weights.gather(0, target)
            focal_weight = focal_weight * alpha.unsqueeze(1)
        
        loss = -focal_weight * log_probs.gather(1, target.unsqueeze(1))
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
            
def compute_class_weights(labels, num_classes):
    """
    
    Args:
        labels: class labels
        num_classes: number of classes
    
    Returns:
        class_weights: weights for each class
    """
    # calculate the number of samples for each class
    counts = torch.zeros(num_classes, device=labels.device)
    for i in range(num_classes):
        counts[i] = (labels == i).sum().float()
    
    counts = counts.clamp(min=1.0)
    
    freqs = counts / counts.sum()

    class_weights = 1.0 / freqs
    
    # normalize weights
    class_weights = class_weights * (num_classes / class_weights.sum())
    
    return class_weights
