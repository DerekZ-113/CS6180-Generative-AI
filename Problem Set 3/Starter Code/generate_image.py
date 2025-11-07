import torch
import utils as ut
from gmvae import GMVAE
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gmvae = GMVAE(z_dim=10, k=500, name='model=gmvae_z=10_k=500_run=0000').to(device)

ut.load_model_by_name(gmvae, global_step=20000, device=device)

with torch.no_grad():
    samples = gmvae.sample_x(200) 
    samples = samples.view(200, 1, 28, 28) 

save_image(samples, 'generated_digits.png', nrow=20)