import torch
from torchvision.transforms import Normalize

# CIFAR10 dataset unnormalize as it comes out of the iter
def unnormalize_cifar10(normed):

    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.201])

    unnormalize = Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    a = unnormalize(normed)
    a = a.transpose(0,1)
    a = a.transpose(1,2)
    a = a * 255
    b = a.clone().detach().requires_grad_(True).type(torch.uint8)
    
    
    
    return b


def unnormalize_femnist(normed):
    mean = torch.tensor([0.1307])
    std = torch.tensor([0.3081])
    
    unnormalize = Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    a = unnormalize(normed)
    b = a.clone().detach().requires_grad_(True)
    return b