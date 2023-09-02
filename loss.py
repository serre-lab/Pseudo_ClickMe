"""
Harmonization loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyramid import pyramidal_representation
from utils import compute_human_alignment

def standardize_cut(heatmaps, axis=(2, 3), epsilon=1e-5):
    """
    Standardize the heatmaps (zero mean, unit variance) and apply ReLU.

    Parameters
    ----------
    heatmaps : torch.Tensor # The heatmaps to standardize. (N, 1, H, W)
    axes : tuple            # The axes to compute the mean and variance.
    epsilon : float         # A small value to avoid division by zero.

    Returns
    -------
    torch.Tensor            # The positive part of the standardized heatmaps.
    """

    means = torch.mean(heatmaps, dim=axis, keepdim=True)
    stds = torch.std(heatmaps, dim=axis, keepdim=True)
    heatmaps = (heatmaps - means) / (stds + epsilon)

    heatmaps = torch.relu(heatmaps)
    return heatmaps

def mse(heatmaps_a, heatmaps_b):
    """
    Compute the Mean Squared Error between two set of heatmaps

    Parameters
    ----------
    heatmap_a : torch.Tensor # The first heatmap.
    heatmap_b : torch.Tensor # The second heatmap.

    Returns
    -------
    torch.Tensor             # The MSE .
    """
    return torch.mean(torch.square(heatmaps_a - heatmaps_b))


def pyramidal_mse(true_heatmaps, predicted_heatmaps, nb_levels=5):
    """
    Compute mean squared error between two set heatmaps on a pyramidal representation.

    Parameters
    ----------
    true_heatmaps : torch.Tensor      # The true heatmaps. (N, 1, H, W)
    predicted_heatmaps : torch.Tensor # The predicted heatmaps. (N, 1, H, W)
    nb_levels : int                   # The number of levels to use in the pyramid.

    Returns
    -------
    torch.Tensor                      # The weighted MSE.

    """
    pyramid_y = pyramidal_representation(true_heatmaps, nb_levels)
    pyramid_y_pred = pyramidal_representation(predicted_heatmaps, nb_levels)
    
    # for i in range(nb_levels):
    #     print(mse(pyramid_y[i], pyramid_y_pred[i]))

    loss = torch.mean(torch.stack(
        [mse(pyramid_y[i], pyramid_y_pred[i]) for i in range(nb_levels)]))

    return loss

def harmonizer_loss(model, images, labels, clickme_maps,
                    criterion, lambda_weights=1e-5, lambda_harmonization=1.0):
    """
    Compute the harmonization loss: cross entropy + pyramidal mse of standardized-cut heatmaps.

    Parameters
    ----------
    model : torch.nn.Module                   # The model to train.
    images : torch.Tensor                     # The batch of images to train on.
    labels : torch.Tensor                     # The batch of labels.
    clickme_maps : torch.Tensor               # The batch of true heatmaps (e.g Click-me maps) to align the model on.
    criterion : torch.nn.CrossEntropyLoss     # The cross entropy loss to use.

    Returns
    -------
    harmonization_loss : torch.Tensor                 
    cce_loss : torch.Tensor   
    """

    model.train()
       
    # compute prediction
    images.requires_grad = True
    output = model(images)

    # get correct class scores
    correct_class_scores = output.gather(1, labels.view(-1, 1)).squeeze()
    device = images.device
    ones_tensor = torch.ones(correct_class_scores.shape).to(device) # scores is a tensor here, need to supply initial gradients of same tensor shape as scores.
    correct_class_scores.backward(ones_tensor, retain_graph=True) # compute the gradients while retain the graph
    
    # obtain saliency map
    grads = torch.abs(images.grad)
    saliency_maps, _ = torch.max(grads, dim=1, keepdim=True) # (N, C, H, W) -> (N, 1, H, W)
    images.grad.zero_() # reset the gradients
    
    # apply the standardization-cut procedure on heatmaps
    saliency_maps = standardize_cut(saliency_maps.detach())
    clickme_maps = standardize_cut(clickme_maps)
    
    # re-normalize before pyramidal
    saliency_max = torch.amax(saliency_maps, (2,3), keepdims=True) + 1e-6
    clickme_max = torch.amax(clickme_maps, (2,3), keepdims=True) + 1e-6
    
    # normalize the true heatmaps according to the saliency maps
    clickme_maps = clickme_maps / clickme_max * saliency_max
    
    # Compute and combine the losses
    pyramidal_loss = pyramidal_mse(saliency_maps, clickme_maps)
    cce_loss = criterion(output, labels)
    # weight_loss = lambda_weights * torch.norm(model.parameters(), 2)**2 # weight_decay in optimizer

    harmonization_loss = cce_loss + pyramidal_loss * lambda_harmonization # + weight_loss

    return harmonization_loss, cce_loss

def harmonization_eval(model, images, labels, clickme_maps, criterion):
    """
    Compute the harmonization loss: cross entropy + pyramidal mse of standardized-cut heatmaps.

    Parameters
    ----------
    model : torch.nn.Module                   # The model to train.
    images : torch.Tensor                     # The batch of images to train on.
    labels : torch.Tensor                     # The batch of labels.
    clickme_maps : torch.Tensor               # The batch of true heatmaps (e.g Click-me maps) to align the model on.
    criterion : torch.nn.CrossEntropyLoss     # The cross entropy loss to use.

    Returns
    -------
    alignment_score : torch.Tensor            # The score to indicate the spearman correlation between saliency maps and clickme maps
    
    """
    model.eval()
    
    # compute prediction and loss
    images.requires_grad = True
    output = model(images)
    cce_loss = criterion(output, labels)
    
    # get correct class scores
    correct_class_scores = output.gather(1, labels.view(-1, 1))#.squeeze()
    device = images.device
    ones_tensor = torch.ones(correct_class_scores.shape).to(device) # scores is a tensor here, need to supply initial gradients of same tensor shape as scores.
    correct_class_scores.backward(ones_tensor, retain_graph=True) # compute the gradients while retain the graph
    
    # compute saliency maps
    grads = torch.abs(images.grad)
    saliency_maps, _ = torch.max(grads, dim=1, keepdim=True) # saliency map (N, C, H, W) -> (N, 1, H, W)
    
    # measure human alignment
    human_alignment = compute_human_alignment(saliency_maps, clickme_maps)
    images.grad.zero_() # reset the gradients
    
    return output, cce_loss, human_alignment

if __name__ == "__main__":
    hmp_a, hmp_b = torch.rand(4, 1, 224, 224), torch.rand(4, 1, 224, 224)
    print(standardize_cut(hmp_a).shape)
    print(mse(hmp_a, hmp_b))
    print(pyramidal_mse(hmp_a, hmp_b))
