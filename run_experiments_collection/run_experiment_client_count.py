"""Run Experiment

This script allows to run one federated learning experiment; the experiment name, the method and the
number of clients/tasks should be precised along side with the hyper-parameters of the experiment.

The results of the experiment (i.e., training logs) are written to ./logs/ folder.

This file can also be imported as a module and contains the following function:

    * run_experiment - runs one experiments given its arguments
"""
from utils.utils import *
from utils.constants import *
from utils.args import *
from run_experiment import * 

from torch.utils.tensorboard import SummaryWriter

# Import General Libraries
import os
import argparse
import torch
import copy
import pickle
import random
import numpy as np
import pandas as pd
from models import *

# Import Transfer Attack
from transfer_attacks.Personalized_NN import *
from transfer_attacks.Params import *
from transfer_attacks.Transferer import *
from transfer_attacks.Args import *
from transfer_attacks.TA_utils import *

import numba 


if __name__ == "__main__":


    exp_names = ['c5','c10','c20','c30','c40']
    adv_mode = True
    
    num_clients_list = [5,10,20,30,40]
    
        
    for itt in range(len(exp_names)):
        
        print("running trial:", itt)
        
        # Manually set argument parameters
        args_ = Args()
        args_.experiment = "cifar10"
        args_.method = 'FedAvg_adv'
        args_.decentralized = False
        args_.sampling_rate = 1.0
        args_.input_dimension = None
        args_.output_dimension = None
        args_.n_learners= 3
        args_.n_rounds = 150
        args_.bz = 128
        args_.local_steps = 1
        args_.lr_lambda = 0
        args_.lr = 0.03
        args_.lr_scheduler = 'multi_step'
        args_.log_freq = 20
        args_.device = 'cuda'
        args_.optimizer = 'sgd'
        args_.mu = 0
        args_.communication_probability = 0.1
        args_.q = 1
        args_.locally_tune_clients = False
        args_.seed = 1234
        args_.verbose = 1
        args_.save_path = 'weights/neurips/cifar/client_count/fedavg_adv_new/' + exp_names[itt]
        args_.validation = False
        args_.save_freq = 20

        # Other Argument Parameters
        Q = 10 # update per round
        G = 0.15
        num_clients = num_clients_list[itt] #50
        S = 0.05 # Threshold
        step_size = 0.01
        K = 10
        eps = 0.1

        # Randomized Parameters
        Ru = np.ones(num_clients)
        
        # Generate the dummy values here
        aggregator, clients = dummy_aggregator(args_, num_clients)

        if adv_mode:
        # Set attack parameters
            x_min = torch.min(clients[0].altered_dataloader.x_data)
            x_max = torch.max(clients[0].altered_dataloader.x_data)


            atk_params = PGD_Params()
            atk_params.set_params(batch_size=1, iteration = K,
                               target = -1, x_val_min = x_min, x_val_max = x_max,
                               step_size = 0.01, step_norm = "inf", eps = eps, eps_norm = 'inf')

            # Obtain the central controller decision making variables (static)
            num_h = args_.n_learners= 3
            Du = np.zeros(len(clients))

            for i in range(len(clients)):
                num_data = clients[i].train_iterator.dataset.targets.shape[0]
                Du[i] = num_data
            D = np.sum(Du) # Total number of data points


        # Train the model
        print("Training..")
        pbar = tqdm(total=args_.n_rounds)
        current_round = 0
        while current_round <= args_.n_rounds:

            
        # For regular em comment here to 
            if adv_mode:
#             # If statement catching every Q rounds -- update dataset
                if  current_round != 0 and current_round%Q == 0: # "ADV Iter"
                    Whu = np.zeros([num_clients,num_h]) # Hypothesis weight for each user
                    for i in range(len(clients)):
                        # print("client", i)
                        temp_client = aggregator.clients[i]
                        hyp_weights = temp_client.learners_ensemble.learners_weights
                        Whu[i] = hyp_weights

                    row_sums = Whu.sum(axis=1)
                    Whu = Whu / row_sums[:, np.newaxis]
                    Wh = np.sum(Whu,axis=0)/num_clients

                    # Solve for adversarial ratio at every client
                    Fu = solve_proportions(G, num_clients, num_h, Du, Whu, S, Ru, step_size)

                    # Assign proportion and attack params
                    # Assign proportion and compute new dataset
                    for i in range(len(clients)):
                        aggregator.clients[i].set_adv_params(Fu[i], atk_params)
                        aggregator.clients[i].update_advnn()
                        aggregator.clients[i].assign_advdataset()
        # here

            aggregator.mix()
            
            if aggregator.c_round != current_round:
                pbar.update(1)
                current_round = aggregator.c_round

        if "save_path" in args_:
            save_root = os.path.join(args_.save_path)

            os.makedirs(save_root, exist_ok=True)
            aggregator.save_state(save_root)
            
        del args_, aggregator, clients
        torch.cuda.empty_cache()
            