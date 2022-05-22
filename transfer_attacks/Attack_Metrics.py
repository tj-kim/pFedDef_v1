import torch
from torch.autograd import Variable


def calcNN_variance(network, data_x, data_y):
    """
    Take in Pytorch nn module (surrogate)
    and data loader tensor data in order to obtain variance of loss across empirical distribution
    """
    
    network.eval()
    
    loss_func = torch.nn.NLLLoss(reduction='none')
    loss = loss_func(network(data_x), data_y)
    EL1 = torch.mean(torch.mul(loss,loss))
    EL2 = torch.mul(torch.mean(loss),torch.mean(loss))
    
    return torch.sub(EL1, EL2)

def calcNN_alignment(network1, network2, data_x, data_y):
    """
    Take in surrogate and victim pytorch nn module, as well as dataloader tensor inputs
    """
    network1.eval()
    network2.eval()
    network1.zero_grad()
    network2.zero_grad()
    
    # Obtain gradient with respect to each input
    x_adv1 = Variable(data_x, requires_grad=True)
    h_adv1 = network1.forward(x_adv1)
    cost1 = network1.criterion(h_adv1, data_y)
    
    x_adv2 = Variable(data_x, requires_grad=True)
    h_adv2 = network2.forward(x_adv2)
    cost2 = network2.criterion(h_adv2, data_y)
    
    if x_adv1.grad is not None:
        x_adv1.grad.data.fill_(0)
    if x_adv2.grad is not None:
        x_adv2.grad.data.fill_(0)
    cost1.backward()
    cost2.backward()
    
    # Loop through each input and calculate norm
    temp = torch.zeros(x_adv1.shape[0])
    for i in range(x_adv1.shape[0]):
        
        # Reshape numerator matrices into vectors to end up with scalar
        g1 = torch.reshape(x_adv1.grad[i,0,:,:],(torch.numel(x_adv1.grad[i,0,:,:]),1))
        g2 = torch.reshape(x_adv2.grad[i,0,:,:],(torch.numel(x_adv2.grad[i,0,:,:]),1))
        
        num = torch.matmul(torch.transpose(g1, 0, 1),g2)
        den = torch.norm(input= x_adv1.grad[i,0,:,:],p=2) * torch.norm(input= x_adv2.grad[i,0,:,:],p=2)
        
        temp[i] = num/den
        
    return torch.acos(torch.mean(temp))

def calcNN_ingrad(network, data_x, data_y, norm=2):
    """
    Take in pytorch nn module (victim)
    and data tensor to obtain size of input gradient 
    """
    
    network.eval()
    
    # Obtain gradient with respect to each input
    x_adv = Variable(data_x, requires_grad=True)
    h_adv = network.forward(x_adv)
    cost = network.criterion(h_adv, data_y)
    
    network.zero_grad()

    if x_adv.grad is not None:
        x_adv.grad.data.fill_(0)
    cost.backward()
    
    # Loop through each input and calculate norm
    temp = torch.zeros(x_adv.shape[0])
    for i in range(x_adv.shape[0]):
        temp[i] = torch.norm(input= x_adv.grad[i,0,:,:],p=norm)
    
    # Find mean of Norms
    return torch.mean(temp)
    