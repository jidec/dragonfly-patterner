import argparse
import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from dcgan import Generator

# code modified from Natsu6767/DCGAN-PyTorch, an implementation of PyTorch's DCGAN tutorial

def ganGenerate(num_images,model_name,proj_dir):
    # Load the checkpoint file.
    # state_dict = torch.load(proj_dir + "/data/ml_models/" + model_name)
    state_dict = torch.load("model/" + model_name)
    # Set the device to run on: GPU or CPU.
    device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
    # Get the 'params' dictionary from the loaded state_dict.
    params = state_dict['params']

    # Create the generator network.
    netG = Generator(params).to(device)
    # Load the trained generator weights.
    netG.load_state_dict(state_dict['generator'])
    print(netG)

    # Get latent vector Z from unit normal distribution.
    noise = torch.randn(int(num_images), params['nz'], 1, 1, device=device)

    # Turn off gradient calculation to speed up the process.
    with torch.no_grad():
        # Get generated image from the noise vector using
        # the trained generator.
        generated_img = netG(noise).detach().cpu()

    # Display the generated image.
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(vutils.make_grid(generated_img, padding=2, normalize=True), (1,2,0)))

    plt.show()