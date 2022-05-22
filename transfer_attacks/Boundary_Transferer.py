# Import General Libraries
import os
import argparse
import torch
import copy
import pickle
import random
import numpy as np
import pandas as pd

# Import FedEM based Libraries
from utils.utils import *
from utils.constants import *
from utils.args import *
from torch.utils.tensorboard import SummaryWriter
from run_experiment import *
from models import *

# Import Transfer Attack
from transfer_attacks.Personalized_NN import *
from transfer_attacks.Params import *
from transfer_attacks.Transferer import *
from transfer_attacks.Args import *

class Boundary_Transferer(): 
    """
    - Load all the datasets but separate them
    - Intermediate values of featues after 2 convolution layers
    """
    
    def __init__(self, models_list, dataloader):
        
        self.models_list = models_list
        self.dataloader = dataloader
        
        self.advNN_idx = None
        self.advNN = None # Dictionary to hold adversary 
        self.atk_order = None
        
        self.base_nn_idx = None
        self.victim_idx = None
        
        self.fixed_point = None # {"idx","x","y"}
        self.comparison_set = None # For legitimate direction
        self.comparison_x = None
        self.comparison_y = None
        
        # Target that is the closest point in legit used in adv targeted attack
        self.y_comparison = None
        
        # Put single model from model_list into adversarial NN (can perform attacks)
        adv_nn = None
        self.atk_params = IFSGM_Params()
        
    def set_adv_NN(self, idx):
        
        self.advNN = copy.deepcopy(Adv_NN(self.models_list[idx], self.dataloader))
        
    def set_ensemble_adv_NN(self, client_idx):
        
        self.advNN_idx = client_idx # List
        self.advNN = {} # Dict of NN
        
        for i in client_idx:
            self.advNN[i] = copy.deepcopy(Adv_NN(self.models_list[i], self.dataloader))
        
        return
        
    def select_data_point(self, explore_set_size = 1000):
        """
        Select a single data point to use as comparison of different boundary types
        Change - all members of the system must classify this point the same 
        """
        self.fixed_point = {}
        found_point_flag = False
        
        while not found_point_flag:
            x, y = self.dataloader.load_batch(explore_set_size)
            correct_idx = {}

            # Analyze Dataset and find point that is classified correctly
            for nidx in range(len(self.models_list)):
                preds = torch.argmax(self.models_list[nidx](x),axis=1)
                correct_idx[nidx] = torch.where(preds == y)[0]

            temp_idx = correct_idx[0]

            for nidx in range(1,len(self.models_list)):
                comp_idx = correct_idx[nidx]
                indices = torch.zeros_like(temp_idx,dtype=torch.bool,device='cuda')
                for elem in comp_idx:
                    indices = indices | (temp_idx==elem)
                temp_idx = temp_idx[indices]

            # Select a point within remaining at random
            # print(temp_idx.numel())
            if not temp_idx.numel():
                continue
            else:
                chosen_idx = temp_idx[np.random.randint(temp_idx.shape[0],size=1)[0]]
                self.fixed_point["x"] = x[chosen_idx]
                self.fixed_point["y"] = y[chosen_idx]
                found_point_flag = True
            
        return 
    
    def select_comparison_set(self,batch_size, og_y = None):
        """
        Select multiple datapoints to use to compare 
        """
        
        xs, ys = self.dataloader.load_batch(batch_size)
        
        self.comparison_set = {}
        
        self.comparison_set["x"] = xs
        self.comparison_set["y"] = ys
        
        # Eliminate data points that are not of same class
        if og_y != None:
            idxs = torch.where(self.comparison_set["y"] != og_y)[0]
            self.comparison_set["x"] = self.comparison_set["x"][idxs]
            self.comparison_set["y"] = self.comparison_set["y"][idxs]
        
        return
    
    def measure_distance(self, x1, x2_set):
        
        x2_dist = torch.subtract(x2_set, x1)
        x2_l2 = torch.linalg.norm(x2_dist.flatten(start_dim=1),ord = 2, dim=1)
        
        return x2_dist, x2_l2
    
    def ensemble_attack_order(self):
        
        num_iters = self.atk_params.iteration
        atk_order = []
        
        for t in range(num_iters):
            idx = t%len(self.advNN_idx)
            atk_order += [self.advNN_idx[idx]]
            
        self.atk_order = atk_order
    
    def sweep_victim_distance(self, og_ep, min_dist_unit_vector, ep_granularity, rep_padding, print_res = False):
        """
        Given direction of misclassification, calculate discance needed to cross boundary of victims
        """
        
        # Sweep epilson in unit direction for all NN
        num_sweep = int(np.ceil(og_ep)/ep_granularity)
        # Base NN
        base_ep = 0
        for e in range(1,num_sweep+1):
            
            delta = (min_dist_unit_vector * e * ep_granularity).unsqueeze(0)
            x_in = self.fixed_point["x"] + delta
            
            y_dist = self.models_list[self.base_nn_idx](x_in)

            y_idx = torch.argmax(y_dist, axis=1)
            
            if y_idx != self.fixed_point["y"]:
                if print_res:
                    print("og distance:", og_ep)
                    print("iteration:",e,", ep:", e * ep_granularity)
                    print("true y:", self.fixed_point["y"])
                    print("pred_y:", y_idx)
                    print("relative_y:",self.comparison_y[min_dist_idx])
                    scaled_y = self.models_list[self.base_nn_idx](self.fixed_point["x"] 
                                + (min_dist_unit_vector * num_sweep * ep_granularity).unsqueeze(0))
                    scaled_y = torch.argmax(scaled_y,axis=1)
                    print("scaled_ep_y:",scaled_y)
                base_ep = e*ep_granularity
                break
            
        # Victim NN
        victim_eps = {}
        num_sweep_v = num_sweep + rep_padding
        for v_idx in self.victim_idx:
            for e in range(1, num_sweep_v+1):
                delta = (min_dist_unit_vector * e * ep_granularity).unsqueeze(0)
                x_in = self.fixed_point["x"] + delta

                y_dist = self.models_list[v_idx](x_in)
                y_idx = torch.argmax(y_dist, axis=1)

                if y_idx != self.fixed_point["y"]:
                    victim_eps[v_idx] = e*ep_granularity
                    break
                                                                          
        return base_ep, victim_eps
    
    def legitimate_direction(self, batch_size, ep_granularity = 0.5, rep_padding = 1000, new_point = True,
                            print_res = False):
        """
        Calculate Legitimate Direction for a single point
        """
        
        # Select point of baseline comparison 
        if new_point:
            self.select_data_point()
        
        # Select set of comparison 
        self.select_comparison_set(batch_size, self.fixed_point["y"])
        
        # Calculate X distance
        x_dists, x_dists_l2 = self.measure_distance(self.fixed_point["x"], self.comparison_set["x"])
        
        # Classify all members of comparison set
        y_pred_nn = None
        min_dist_idx = None
        min_dist_unit_vector = None
        
        # Classify each data for each classifier
        temp_classified = self.models_list[self.base_nn_idx](self.comparison_set["x"])
        y_pred_nn = torch.argmax(temp_classified,axis=1)

        # Filter twice - argmin (distance), conditioned on different label
        dist_mask = torch.where(y_pred_nn != self.fixed_point["y"], x_dists_l2, torch.max(x_dists_l2))
        min_dist_idx = torch.argmin(dist_mask)
        min_dist_unit_vector = torch.divide(x_dists[min_dist_idx], 
                                                torch.linalg.norm(x_dists[min_dist_idx].flatten(),ord=2))
        
        og_ep = torch.linalg.norm(x_dists[min_dist_idx].flatten(),ord=2).data.tolist()
            
        self.y_comparison = self.comparison_set["y"][min_dist_idx]
        
        base_ep, victim_eps = self.sweep_victim_distance(og_ep, min_dist_unit_vector, 
                                                         ep_granularity, rep_padding, print_res = False)
                                                                          
        return base_ep, victim_eps
    
    def adversarial_direction(self, ep_granularity = 0.5, rep_padding = 1000, new_point = True, 
                              print_res = False, target = True):
        """
        Calculate adversarial direction for a single point (same as before?) 
        Return base_epsilon for comparison model, and victim_eps for others
        """
        
        # Select point of baseline comparison 
        if new_point:
            self.select_data_point()
        
        # Perform a single step of attack on data point 
        x_in = self.fixed_point["x"]
        y_in = self.fixed_point["y"]
        
        if target and (self.y_comparison is not None):
            self.atk_params.target = self.y_comparison
        
        # 1.1.22 Changed from ifsgm sub to pgd sub)
        # self.advNN.pgd_sub(self.atk_params,x_in.unsqueeze(0),y_in.unsqueeze(0))
        self.advNN.i_fgsm_sub(self.atk_params,x_in.unsqueeze(0),y_in.unsqueeze(0))
        
        x_adv = self.advNN.x_adv
        dist_diff = torch.subtract(x_adv, x_in)
        
        min_dist_unit_vector = torch.divide(dist_diff, torch.linalg.norm(dist_diff.flatten(),ord=2))[0]
        og_ep = torch.linalg.norm(dist_diff.flatten(),ord=2).data.tolist()
        
        base_ep, victim_eps = self.sweep_victim_distance(og_ep, min_dist_unit_vector, 
                                                         ep_granularity, rep_padding, print_res = False)
                                                                                                                                                    
        return base_ep, victim_eps
    
    def ensemble_adversarial_direction(self, ep_granularity = 0.5, rep_padding = 1000, new_point = True, 
                              print_res = False, target = True):
        
        # Select point of baseline comparison 
        if new_point:
            self.select_data_point()
        
        # Perform a single step of attack on data point 
        x_in = self.fixed_point["x"]
        y_in = self.fixed_point["y"]
        
        adv_x_in = x_in.unsqueeze(0)
        adv_y_in = y_in.unsqueeze(0)
        
        if target and (self.y_comparison is not None):
            self.atk_params.target = self.y_comparison
            
        # Decide on attack sequence
        self.ensemble_attack_order()
        
        # Alter number of iteration in params to 1 
        temp_params = copy.deepcopy(self.atk_params)
        temp_params.iteration = 1
        
        for idx in self.atk_order:
            self.advNN[idx].i_fgsm_sub(temp_params, adv_x_in, adv_y_in)
            adv_x_in = copy.deepcopy(self.advNN[idx].x_adv)
        
        # Record relevant tensors
        x_adv = copy.deepcopy(adv_x_in).cuda()
        
        dist_diff = torch.subtract(x_adv, x_in)
        
        min_dist_unit_vector = torch.divide(dist_diff, torch.linalg.norm(dist_diff.flatten(),ord=2))[0]
        og_ep = torch.linalg.norm(dist_diff.flatten(),ord=2).data.tolist()
        
        base_ep, victim_eps = self.sweep_victim_distance(og_ep, min_dist_unit_vector, 
                                                         ep_granularity, rep_padding, print_res = False)
                                                                                                                                                    
        return base_ep, victim_eps