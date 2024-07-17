import torch
import torch.nn.functional as F
import numpy as np 

def compute_pixelwise_mahalanobis_distance_2d(input_image, reconstructions,mask=None,chol=False, regu=1e-2,SSIM=False,bandwidth=0,invert_bw=False):
    """
    Compute the per-pixel Mahalanobis Distance considering spatial dependencies.

    Parameters:
    - input_image: The original image (2D PyTorch tensor [H, W]).
    - reconstructions: A tensor of reconstructed images (3D tensor [num_reconstructions, H, W]).
    - mask: A tensor of the same size as the input image, where 0 indicates a pixel that should be ignored.
    Returns:
    - Anomaly map based on Mahalanobis Distance (2D PyTorch tensor [H, W]).
    """


    # Ensure the tensors are on the same device
    device = input_image.device
    reconstructions = reconstructions.to(device)

    # Flatten the images for covariance calculation
    num_reconstructions, H, W = reconstructions.shape

    reconstructions_flat = reconstructions.view(num_reconstructions, -1)
    input_image_flat = input_image.view(-1)
    if mask is not None:
        # only use pixels > 0 in the mask for the calculation but reshape the mask to a 2D tensor of shape [H*W] afterwards
        mask_flat = mask.reshape(-1)
        reconstructions_flat = reconstructions_flat[:,mask_flat>0]
        input_image_flat = input_image_flat[mask_flat>0]

    
    # Compute the mean and covariance matrix of the reconstructions
    mean_reconstruction_flat = torch.mean(reconstructions_flat, dim=0)
    difference_vector_flat = input_image_flat - mean_reconstruction_flat
    reconstructions_centered = reconstructions_flat - mean_reconstruction_flat

    covariance_matrix = (reconstructions_centered.t() @ reconstructions_centered) / (num_reconstructions - 1)
    # Regularize the covariance matrix and compute its inverse

    regularization_term = regu * torch.eye(covariance_matrix.shape[0], device=device)
    covariance_matrix = covariance_matrix.to(torch.float32) 
    covariance_matrix += regularization_term



    inv_covariance_matrix = torch.inverse(covariance_matrix)
    inv_covariance_matrix = inv_covariance_matrix.to(torch.float16)

    # Compute Mahalanobis Distance for each pixel
    md_values_flat = difference_vector_flat @ inv_covariance_matrix * difference_vector_flat

    if mask is not None:    
        md_map = torch.zeros((H*W),device=device)
        md_map[mask_flat>0] = md_values_flat.sqrt()
        md_map = md_map.view(H, W)
    else:
        md_map = md_values_flat.sqrt().view(H, W)
    
    return md_map


def compute_pixelwise_mahalanobis_distance_slicewise_3d(input_volume, reconstructions,window_size=1,mask=None,regu=1e-2):
    """
    Compute the per-pixel Mahalanobis Distance considering spatial dependencies.
    In the paper, we do use a window size of 1, i.e., no neighborhood information across slices is considered.
    Parameters:
    - input_volume: The original volume (3D PyTorch tensor [D, H, W]).
    - reconstructions: A tensor of reconstructed volumes (4D tensor [num_reconstructions, D, H, W]).

    Returns:
    - Anomaly map based on Mahalanobis Distance (3D PyTorch tensor [D, H, W]).
    """
    
    num_slices = input_volume.shape[0]
    H = input_volume.shape[1]
    W = input_volume.shape[2]

    md_map = torch.zeros_like(input_volume)
    if window_size == 1: # default case
        for i in range(num_slices):
            mask_slice = mask[i,:,:] if mask is not None else None
            md_map[i,:,:] = compute_pixelwise_mahalanobis_distance_2d(input_volume[i,:,:], reconstructions[:,i,:,:],mask=mask_slice,regu=regu)
        return md_map
    else:  # sliding window approach with neighborhood information
        # padding
        padding = window_size//2
        padded_input_volume = torch.zeros((num_slices+2*padding,H,W))
        padded_input_volume[padding:-padding,:,:] = input_volume
        padded_reconstructions = torch.zeros((reconstructions.shape[0],num_slices+2*padding,H,W))
        padded_reconstructions[:,padding:-padding,:,:] = reconstructions

        # process each slice with the sliding window approach (not applied in the paper)
        for i in range(num_slices):
            md_map[i,:,:] = compute_pixelwise_mahalanobis_distance_3d(padded_input_volume[i:i+window_size,:,:], padded_reconstructions[:,i:i+window_size,:,:])[padding,:,:]
        return md_map


        