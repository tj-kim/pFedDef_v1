import torch
import torch.nn as nn

from collections import OrderedDict 
import itertools

import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import csv
import os
import pickle
from torch.autograd import Variable
import copy

# from transfer_attacks.utils import cuda
from transfer_attacks.projected_gradient_descent import *
from transfer_attacks.utils import *

class Personalized_NN(nn.Module):
    """
    Pytorch NN module that combines head and neck weights for a layered based sharing system
    For training federated learning neural network.
    """
    
    def __init__(self, trained_network):
        
        # Init attributes
        super(Personalized_NN, self).__init__()
        self.trained_network = trained_network
        self.criterion = nn.NLLLoss()
        self.cuda = True
        
        # test_acc attributes
        self.orig_test_acc = None
        self.adv_test_acc = None
        
        # Split Test att
        self.orig_test_acc_robust = None
        self.orig_output_sim_robust = None
        self.orig_test_acc_adv = None
        self.orig_output_sim_adv = None
        
        self.orig_output_sim = None
        self.adv_output_sim = None
        
        self.orig_target_achieve = None
        self.adv_target_achieve = None
        
        # Log correct vs. Incorrect Indices
        self.adv_indices = None # tensor List of indices from batch that are fooled
        self.robust_indices = None # List of indices from batch robust
        
    def forward(self,x):
        
        if torch.cuda.is_available():
            x = x.cuda()
        
        x = self.trained_network.forward(x)
        
        return x
    
    def ensemble_forward_transfer(self, x_orig, x_adv, true_labels, target):
        """
        Same function as forward transfer with less calculations
        """
        self.eval()
        
        batch_size = x_orig.shape[0]
        
        # Forward Two Input Types
        h_adv = self.forward(x_adv)
        h_orig = self.forward(x_orig)
        h_adv_category = torch.argmax(h_adv,dim = 1)
        h_orig_category = torch.argmax(h_orig,dim = 1)
        
        # Record Different Parameters
        self.orig_target_achieve = (h_orig_category == target).float().sum()/batch_size
        self.adv_target_achieve = (h_adv_category == target).float().sum()/batch_size
            
        return (h_adv, h_orig, h_adv_category, h_orig_category)
        
    
    def forward_transfer(self, x_orig, x_adv, y_orig, y_adv,
                         true_labels, target, transfer_diag = True, print_info = False):
        """
        Assume that input images are in pytorch tensor format
        """
        self.eval()
        
        # Cuda Availability
        if torch.cuda.is_available():
            (y_orig, y_adv) = (y_orig.cuda(), y_adv.cuda())
        
        batch_size = y_orig.shape[0]
        
        # Forward Two Input Types
        h_adv = self.forward(x_adv)
        h_orig = self.forward(x_orig)
        h_adv_category = torch.argmax(h_adv,dim = 1)
        h_orig_category = torch.argmax(h_orig,dim = 1)
        
        # Record Different Parameters
        self.orig_test_acc = (h_orig_category == true_labels).float().sum()/batch_size
        self.adv_test_acc = (h_adv_category == true_labels).float().sum()/batch_size # alter
        
        self.orig_output_sim = (h_orig_category == y_orig).float().sum()/batch_size
        self.adv_output_sim = (h_adv_category == y_adv).float().sum()/batch_size
        
        self.orig_target_achieve = (h_orig_category == target).float().sum()/batch_size
        self.adv_target_achieve = (h_adv_category == target).float().sum()/batch_size # alter
        
        if transfer_diag and target > -1:
            # adv test acc
            true_label_idx = true_labels != target
            h_adv_conditioned = h_adv_category[true_label_idx]
            true_labels_conditioned = true_labels[true_label_idx]
            new_batch_size = true_labels_conditioned.shape[0]
            self.adv_test_acc = (h_adv_conditioned == true_labels_conditioned).float().sum()/new_batch_size
            
            # adv target achieve
            self.adv_target_achieve = (h_adv_conditioned == target).float().sum()/new_batch_size
        
        # Record based on indices
        self.adv_indices = h_adv_category == target
        self.robust_indices = h_adv_category != target
        
        # Record split losses
        self.orig_test_acc_robust = (h_orig_category[self.robust_indices] == true_labels[self.robust_indices]).float().sum()/h_orig_category[self.robust_indices].shape[0]
        self.orig_output_sim_robust = (h_orig_category[self.robust_indices] == y_orig[self.robust_indices]).float().sum()/h_orig_category[self.robust_indices].shape[0]
        
        self.orig_test_acc_adv = (h_orig_category[self.adv_indices] == true_labels[self.adv_indices]).float().sum()/h_orig_category[self.adv_indices].shape[0]
        self.orig_output_sim_adv = (h_orig_category[self.adv_indices] == y_orig[self.adv_indices]).float().sum()/h_orig_category[self.adv_indices].shape[0]
            

        # Print Relevant Information
        if print_info:
            print("---- Attack Transfer:", "----\n")
            print("         Orig Test Acc:", self.orig_test_acc.item())
            print("          Adv Test Acc:", self.adv_test_acc.item())
            print("Orig Output Similarity:", self.orig_output_sim.item())
            print(" Adv Output Similarity:", self.adv_output_sim.item())
            print("       Orig Target Hit:", self.orig_target_achieve.item())
            print("        Adv Target Hit:", self.adv_target_achieve.item())
            
        return (h_adv, h_orig, h_adv_category, h_orig_category)


class Adv_NN(Personalized_NN):
    
    def __init__(self, trained_network, dataloader):
        
        # Init attributes
        super(Adv_NN, self).__init__(trained_network)
        
        self.dataloader = dataloader
        
        # Attack outputs
        self.x_orig = None
        self.x_adv = None
        self.y_orig = None
        
        self.softmax_orig = None
        self.output_orig = None
        self.softmax_adv = None
        self.output_adv = None 
        
        
    def i_fgsm_sub(self, atk_params, x_in, y_in):
        """
        Sub-problem of running 
        """
        
        self.eval()
        self.x_adv = Variable(x_in, requires_grad=True)
        
        target= atk_params.target
        eps= atk_params.eps
        alpha= atk_params.alpha
        iteration= atk_params.iteration
        x_val_min= atk_params.x_val_min
        x_val_max= atk_params.x_val_max
        
        for i in range(iteration):
            
            h_adv = self.forward(self.x_adv)
            
            # Loss function based on target
            if target > -1:
                target_tensor = torch.LongTensor(y_in.size()).fill_(target)
                target_tensor = Variable(cuda(target_tensor, self.cuda), requires_grad=False)
                cost = self.criterion(h_adv, target_tensor)
            else:
                cost = -self.criterion(h_adv, y_in)

            self.zero_grad()

            if self.x_adv.grad is not None:
                self.x_adv.grad.data.fill_(0)
            cost.backward()

            self.x_adv.grad.sign_()
            self.x_adv = self.x_adv - alpha*self.x_adv.grad
            self.x_adv = where(self.x_adv > x_in+eps, x_in+eps, self.x_adv)
            self.x_adv = where(self.x_adv < x_in-eps, x_in-eps, self.x_adv)
            self.x_adv = torch.clamp(self.x_adv, x_val_min, x_val_max)
            self.x_adv = Variable(self.x_adv.data, requires_grad=True)
            
        return 
    
    def pgd_sub(self, atk_params, x_in, y_in, x_base = None):
        """
        Perform PGD without post-attack analysis
        """
        self.eval()
        
        # Import attack parameters
        eps_norm = atk_params.eps_norm
        batch_size = atk_params.batch_size
        target= atk_params.target
        eps= atk_params.eps
        alpha= atk_params.step_size
        iteration= atk_params.iteration
        x_val_min= atk_params.x_val_min
        x_val_max= atk_params.x_val_max
        
        # Load data to perturb
    
        self.x_orig  = x_in
        self.y_orig = y_in
        
        if torch.cuda.is_available():
            self.y_orig = self.y_orig.cuda()
        
        self.target = target
        
        # Add random noise within norm ball for start (FOR BATCH)
        noise_unscaled = torch.rand(self.x_orig.shape)
        if eps_norm == 'inf':
            # Workaround as PyTorch doesn't have elementwise clip
            norm_scale = torch.ones_like(noise_unscaled[0]).norm(float('inf'))
        else:
            norm_scale = torch.ones_like(noise_unscaled[0]).norm(eps_norm)                                                          
        noise_scaled = (noise_unscaled*eps/norm_scale).cuda()
        
        self.x_adv = Variable(self.x_orig+noise_scaled, requires_grad=True)
        
        for i in range(iteration):
            
            h_adv = self.forward(self.x_adv)
            
            # Loss function based on target
            if target > -1:
                target_tensor = torch.LongTensor(self.y_orig.size()).fill_(target)
                target_tensor = Variable(cuda(target_tensor, self.cuda), requires_grad=False)
                cost = self.criterion(h_adv, target_tensor)
            else:
                cost = -self.criterion(h_adv, self.y_orig)

            self.zero_grad()

            if self.x_adv.grad is not None:
                self.x_adv.grad.data.fill_(0)
            cost.backward()

            self.x_adv.grad.sign_()
            self.x_adv = self.x_adv - alpha*self.x_adv.grad
            
            
            if eps_norm == 'inf':
                # Workaround as PyTorch doesn't have elementwise clip
                self.x_adv = torch.max(torch.min(self.x_adv, self.x_orig + eps), self.x_orig - eps)
            else: # Other norms (mostly 2 norm)
                delta = self.x_adv - self.x_orig

                # Assume x and x_adv are batched tensors where the first dimension is
                # a batch dimension
                mask = delta.view(delta.shape[0], -1).norm(eps_norm, dim=1) <= eps

                scaling_factor = delta.view(delta.shape[0], -1).norm(eps_norm, dim=1)
                scaling_factor[mask] = eps

                # .view() assumes batched images as a 4D Tensor
                delta *= eps / scaling_factor.view(-1, 1, 1, 1)

                self.x_adv = self.x_orig + delta
            
            self.x_adv = torch.clamp(self.x_adv, x_val_min, x_val_max)
            self.x_adv = Variable(self.x_adv.data, requires_grad=True)
        
        return
        
    def pgd(self, atk_params, print_info=False, mode='test'):
        
        self.eval()
        
        # Import attack parameters
        eps_norm = atk_params.eps_norm
        batch_size = atk_params.batch_size
        target= atk_params.target
        eps= atk_params.eps
        alpha= atk_params.step_size
        iteration= atk_params.iteration
        x_val_min= atk_params.x_val_min
        x_val_max= atk_params.x_val_max
        
        # Load data to perturb
    
        data_x, data_y = self.dataloader.load_batch(batch_size, mode=mode)
        
        self.x_orig  = data_x.reshape(batch_size, data_x.shape[1],data_x.shape[2],data_x.shape[3])
        self.y_orig = data_y.type(torch.LongTensor)
        
        if torch.cuda.is_available():
            self.y_orig = self.y_orig.cuda()
        
        self.target = target
        
        # Add random noise within norm ball for start (FOR BATCH)
        noise_unscaled = torch.rand(self.x_orig.shape)
        if eps_norm == 'inf':
            # Workaround as PyTorch doesn't have elementwise clip
            norm_scale = torch.ones_like(noise_unscaled[0]).norm(float('inf'))
        else:
            norm_scale = torch.ones_like(noise_unscaled[0]).norm(eps_norm)                                                          
        noise_scaled = (noise_unscaled*eps/norm_scale).cuda()
        
        self.x_adv = Variable(self.x_orig+noise_scaled, requires_grad=True)
        
        for i in range(iteration):
            
            h_adv = self.forward(self.x_adv)
            
            # Loss function based on target
            if target > -1:
                target_tensor = torch.LongTensor(self.y_orig.size()).fill_(target)
                target_tensor = Variable(cuda(target_tensor, self.cuda), requires_grad=False)
                cost = self.criterion(h_adv, target_tensor)
            else:
                cost = -self.criterion(h_adv, self.y_orig)

            self.zero_grad()

            if self.x_adv.grad is not None:
                self.x_adv.grad.data.fill_(0)
            cost.backward()

            self.x_adv.grad.sign_()
            self.x_adv = self.x_adv - alpha*self.x_adv.grad
            
            
            if eps_norm == 'inf':
                # Workaround as PyTorch doesn't have elementwise clip
                self.x_adv = torch.max(torch.min(self.x_adv, self.x_orig + eps), self.x_orig - eps)
            else: # Other norms (mostly 2 norm)
                delta = self.x_adv - self.x_orig

                # Assume x and x_adv are batched tensors where the first dimension is
                # a batch dimension
                mask = delta.view(delta.shape[0], -1).norm(eps_norm, dim=1) <= eps

                scaling_factor = delta.view(delta.shape[0], -1).norm(eps_norm, dim=1)
                scaling_factor[mask] = eps

                # .view() assumes batched images as a 4D Tensor
                delta *= eps / scaling_factor.view(-1, 1, 1, 1)

                self.x_adv = self.x_orig + delta
            
            self.x_adv = torch.clamp(self.x_adv, x_val_min, x_val_max)
            self.x_adv = Variable(self.x_adv.data, requires_grad=True)
        
        self.post_attack(batch_size = batch_size, print_info = print_info)
        
        return
        
    def i_fgsm(self, atk_params, print_info=False, mode = 'test'):
        """
            Perform IFSGM attack on a randomly sampled batch 
            All attack params and batch sizes are defiend in atk_params
        """
        self.eval()
        
        # Import attack parameters
        batch_size = atk_params.batch_size
        target= atk_params.target
        eps= atk_params.eps
        alpha= atk_params.alpha
        iteration= atk_params.iteration
        x_val_min= atk_params.x_val_min
        x_val_max= atk_params.x_val_max
        
        # Load data to perturb
    
        data_x, data_y = self.dataloader.load_batch(batch_size, mode=mode)
        self.x_orig  = data_x.reshape(batch_size,3,32,32)
        self.y_orig = data_y.type(torch.LongTensor)
        
        if torch.cuda.is_available():
            self.y_orig = self.y_orig.cuda()
        
        self.target = target
        
        self.x_adv = Variable(self.x_orig, requires_grad=True)
        
        for i in range(iteration):
            
            h_adv = self.forward(self.x_adv)
            
            # Loss function based on target
            if target > -1:
                target_tensor = torch.LongTensor(self.y_orig.size()).fill_(target)
                target_tensor = Variable(cuda(target_tensor, self.cuda), requires_grad=False)
                cost = self.criterion(h_adv, target_tensor)
            else:
                cost = -self.criterion(h_adv, self.y_orig)

            self.zero_grad()

            if self.x_adv.grad is not None:
                self.x_adv.grad.data.fill_(0)
            cost.backward()

            self.x_adv.grad.sign_()
            self.x_adv = self.x_adv - alpha*self.x_adv.grad
            self.x_adv = where(self.x_adv > self.x_orig+eps, self.x_orig+eps, self.x_adv)
            self.x_adv = where(self.x_adv < self.x_orig-eps, self.x_orig-eps, self.x_adv)
            self.x_adv = torch.clamp(self.x_adv, x_val_min, x_val_max)
            self.x_adv = Variable(self.x_adv.data, requires_grad=True)

        self.post_attack(batch_size = batch_size, print_info = print_info)
            
    def post_attack(self, batch_size, print_info = False):
        """
        Computes attack success metrics after xadv is generated
        """
                
        self.softmax_orig = self.forward(self.x_orig)
        self.output_orig = torch.argmax(self.softmax_orig,dim=1)
        self.softmax_adv = self.forward(self.x_adv)
        self.output_adv = torch.argmax(self.softmax_adv,dim=1)
        
        # Record accuracy and loss
        self.orig_loss = self.criterion(self.softmax_orig, self.y_orig).item()
        self.adv_loss = self.criterion(self.softmax_adv, self.y_orig).item()
        self.orig_acc = (self.output_orig == self.y_orig).float().sum()/batch_size
        self.adv_acc = (self.output_adv == self.y_orig).float().sum()/batch_size
        
        # Add Perturbation Distance (L2 norm) - across each input
        self.norm = torch.norm(torch.sub(self.x_orig, self.x_adv, alpha=1),dim=(2,3))

        # Print Relevant Information
        if print_info:
            print("---- FGSM Batch Size:", batch_size, "----\n")
            print("Orig Target:", self.y_orig.tolist())
            print("Orig Output:", self.output_orig.tolist())
            print("ADV Output :", self.output_adv.tolist(),'\n')
            print("Orig Loss  :", self.orig_loss)
            print("ADV Loss   :", self.adv_loss,'\n')
            print("Orig Acc   :", self.orig_acc.item())
            print("ADV Acc    :", self.adv_acc.item())