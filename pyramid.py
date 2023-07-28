import torch
import torch.nn.functional as F
import time

"""
Reference: 
-   PyTorch Implementation of Laplacian Pyramid Loss
    https://gist.github.com/Harimus/918fddd8bdc6e13e4acf3d213f2f24cd
"""

def downsample(image, kernel):
    if len(image.shape) != 4:
        raise IndexError('Expected input tensor to be of shape: (batch, depth, height, width) but got: ' + str(x.shape))
    
    groups = kernel.shape[1]
    padding = kernel.shape[-1] // 2 # Kernel size needs to be odd number
    
    # x = F.conv2d(image, weight=kernel, stride=1, padding=padding, groups=channels)
    # return x[:, :, ::2, ::2]
    
    x = F.conv2d(image, weight=kernel, stride=2, padding=padding, groups=groups)
    return x

def gaussian_kernel(num_channels):
    
    # kernel = np.array((1., 4., 6., 4., 1.), dtype=np.float32)
    # kernel = np.outer(kernel, kernel)
    # kernel /= np.sum(kernel)
    # kernel = torch.tensor(kernel, dtype=torch.float32)
    
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]]) # 5 * 5 Gaussian Kernel
    kernel /= 256.0 # torch.sum(kernel)
    return kernel.repeat(num_channels, 1, 1, 1) # (C, output dim, H, W)

def pyramidal_representation(image, num_levels):
    """
    Compute the pyramidal representation of an image.

    Parameters
    ----------
    image : torch.Tensor (N, C, H, W)
        The image to compute the pyramidal representation.
    num_levels : int
        The number of levels to use in the pyramid.

    Returns
    -------
    list of torch.Tensor
        The pyramidal representation.
    """
    device = image.device
    kernel = gaussian_kernel(image.shape[1]).to(device)
    levels = [image]
    for _ in range(num_levels):
        image = downsample(image, kernel)
        levels.append(image)
    return levels

if __name__ == '__main__':
    img = torch.rand(64, 1, 128, 128)
    reps = pyramidal_representation(img, 5)

    for rep in reps:
        print(rep.shape)
        
    '''
    https://gist.github.com/patrickmineault/21b8d78f423ac8ea4b006f9ec1a1a1a7
    '''
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]]).reshape(1, 1, 5, 5)
    kernel = kernel.repeat((img.shape[1], 1, 1, 1))

    # Downsample by a factor 2 with smoothing
    start = time.time()
    padding = kernel.shape[-1] // 2
    sz = 224
    for i in range(5):
        sz /= 2
        mask = torch.ones(1, *img.shape[1:])
        mask = F.conv2d(mask, kernel, groups=img.shape[1], stride=2, padding=padding)
        img = F.conv2d(img, kernel, groups=img.shape[1], stride=2, padding=padding)
        img = img / mask # Normalize the edges and corners.
        print(img.shape)
    end = time.time()
    print(end - start)
    
    
    
    
    
