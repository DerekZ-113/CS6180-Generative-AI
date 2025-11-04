import torch
from torch.nn import functional as F

def loss_nonsaturating(g, d, x_real, *, device):
    '''
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): nonsaturating discriminator loss
    - g_loss (torch.Tensor): nonsaturating generator loss
    '''
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)

    # YOUR CODE STARTS HERE
    # You may find some or all of the below useful:
    #   - F.binary_cross_entropy_with_logits
    #   - F.logsigmoid
    raise NotImplementedError
    # YOUR CODE ENDS HERE

    return d_loss, g_loss

