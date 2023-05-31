# Import General Libraries
import os
import argparse
import torch
import copy
import pickle
import random
import numpy as np
import pandas as pd
import sys

# Import FedEM based Libraries
from utils.utils import *
from utils.constants import *
from utils.args import *
from run_experiment import *
from models import *

# Import Transfer Attack
from transfer_attacks.Personalized_NN import *
from transfer_attacks.Params import *
from transfer_attacks.Transferer import *
from transfer_attacks.Args import *
from transfer_attacks.TA_utils import *
from transfer_attacks.Boundary_Transferer import *

## ALL_INPUTS + HELPER ##
all_inputs = np.genfromtxt(sys.argv[1], dtype = str, delimiter = "\t")
def revised_input(input):
    return str.split(input, "= ")[1]

setting = revised_input(all_inputs[0])
if setting == 'FedEM':
    nL = 3
else:
    nL = 1

## PREPARE OUTPUT FILE ##
output_file = open(sys.argv[2], "w")

# Manually set argument parameters
args_ = Args()
args_.experiment = revised_input(all_inputs[1])
args_.method = setting
args_.decentralized = eval(revised_input(all_inputs[2]))
args_.sampling_rate = float(revised_input(all_inputs[3]))
args_.input_dimension = None if revised_input(all_inputs[4]) == "None" else revised_input(all_inputs[4])
args_.output_dimension = None if revised_input(all_inputs[5]) == "None" else revised_input(all_inputs[5])
args_.n_learners= nL
args_.n_rounds = int(revised_input(all_inputs[6]))
args_.bz = int(revised_input(all_inputs[7]))
args_.local_steps = int(revised_input(all_inputs[8]))
args_.lr_lambda = int(revised_input(all_inputs[9]))
args_.lr = float(revised_input(all_inputs[10]))
args_.lr_scheduler = revised_input(all_inputs[11])
args_.log_freq = int(revised_input(all_inputs[12]))
args_.device = revised_input(all_inputs[13])
args_.optimizer = revised_input(all_inputs[14])
args_.mu = int(revised_input(all_inputs[15]))
args_.communication_probability = float(revised_input(all_inputs[16]))
args_.q = int(revised_input(all_inputs[17]))
args_.locally_tune_clients = eval(revised_input(all_inputs[18]))
args_.seed = int(revised_input(all_inputs[19]))
args_.verbose = int(revised_input(all_inputs[20]))
args_.save_path = revised_input(all_inputs[21])
args_.validation = eval(revised_input(all_inputs[22]))

# Generate the dummy values here
aggregator, clients = dummy_aggregator(args_, num_user=40)

# Compiling Dataset from Clients
# Combine Validation Data across all clients as test
data_x = []
data_y = []

for i in range(1):
    daniloader = clients[i].test_iterator
    for (x,y,idx) in daniloader.dataset:
        data_x.append(x)
        data_y.append(y)

data_x = torch.stack(data_x)
try:
    data_y = torch.stack(data_y)        
except:
    data_y = torch.FloatTensor(data_y) 
    
dataloader = Custom_Dataloader(data_x, data_y)

# Import Model Weights
num_models = 40

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

if setting == 'local':

#     args_.save_path = 'weights/final/femnist/fig1_take3/local_benign/'
    aggregator.load_state(args_.save_path)
    
    model_weights = []
#     weights = np.load("weights/final/femnist/fig1_take3/local_benign/train_client_weights.npy")
    weights = np.load(args_.save_path+"/train_client_weights.npy")
    
    for i in range(num_models):
        model_weights += [weights[i]]

    # Generate the weights to test on as linear combinations of the model_weights
    models_test = []

    for i in range(num_models):
        new_model = copy.deepcopy(aggregator.clients[i].learners_ensemble.learners[0].model)
        new_model.eval()
        models_test += [new_model]

elif setting == 'FedAvg':
    
#     args_.save_path = 'weights/final/femnist/fig1_take3/fedavg_benign/'
    aggregator.load_state(args_.save_path)
    
    # This is where the models are stored -- one for each mixture --> learner.model for nn
    hypotheses = aggregator.global_learners_ensemble.learners

    # obtain the state dict for each of the weights 
    weights_h = []

    for h in hypotheses:
        weights_h += [h.model.state_dict()]

#     weights = np.load("weights/final/femnist/fig1_take3/fedavg_benign/train_client_weights.npy")
    weights = np.load(args_.save_path+"/train_client_weights.npy")
    
    # Set model weights
    model_weights = []

    for i in range(num_models):
        model_weights += [weights[i]]

    # Generate the weights to test on as linear combinations of the model_weights
    models_test = []

    for (w0) in model_weights:
        # first make the model with empty weights
        new_model = copy.deepcopy(hypotheses[0].model)
        new_model.eval()
        new_weight_dict = copy.deepcopy(weights_h[0])
        for key in weights_h[0]:
            new_weight_dict[key] = w0[0]*weights_h[0][key] 
        new_model.load_state_dict(new_weight_dict)
        models_test += [new_model]

elif setting == 'FedEM':
    
#     args_.save_path = 'weights/final/femnist/fig1_take3/fedem_benign/'
#     args_.save_path = 'weights/final/femnist/fig1_take3/fedem_adv/'
    aggregator.load_state(args_.save_path)
    
    # This is where the models are stored -- one for each mixture --> learner.model for nn
    hypotheses = aggregator.global_learners_ensemble.learners

    # obtain the state dict for each of the weights 
    weights_h = []

    for h in hypotheses:
        weights_h += [h.model.state_dict()]

#     weights = np.load("weights/final/femnist/fig1_take3/fedem_benign/train_client_weights.npy")
#     weights = np.load("weights/final/femnist/fig1_take3/fedem_adv/train_client_weights.npy")
    weights = np.load(args_.save_path+"/train_client_weights.npy")

    # Set model weights
    model_weights = []

    for i in range(num_models):
        model_weights += [weights[i]]


    # Generate the weights to test on as linear combinations of the model_weights
    models_test = []

    for (w0,w1,w2) in model_weights:
        # first make the model with empty weights
        new_model = copy.deepcopy(hypotheses[0].model)
        new_model.eval()
        new_weight_dict = copy.deepcopy(weights_h[0])
        for key in weights_h[0]:
            new_weight_dict[key] = w0*weights_h[0][key] + w1*weights_h[1][key] + w2*weights_h[2][key]
        new_model.load_state_dict(new_weight_dict)
        models_test += [new_model]

        # Here we will make a dictionary that will hold results
logs_adv = []

for i in range(num_models):
    adv_dict = {}
    adv_dict['orig_acc_transfers'] = None
    adv_dict['orig_similarities'] = None
    adv_dict['adv_acc_transfers'] = None
    adv_dict['adv_similarities_target'] = None
    adv_dict['adv_similarities_untarget'] = None
    adv_dict['adv_target'] = None
    adv_dict['adv_miss'] = None
    adv_dict['metric_alignment'] = None
    adv_dict['ib_distance_legit'] = None
    adv_dict['ib_distance_adv'] = None

    logs_adv += [adv_dict]

# Perform transfer attack from one client to another and record stats

# Run Measurements for both targetted and untargeted analysis
new_num_models = len(models_test)
victim_idxs = range(new_num_models)
custom_batch_size = 500
eps = 4.5


for adv_idx in victim_idxs:
    print("\t Adv idx:", adv_idx)
    
    dataloader = load_client_data(clients = clients, c_id = adv_idx, mode = 'test') # or test/train
    
    batch_size = min(custom_batch_size, dataloader.y_data.shape[0])
    
    t1 = Transferer(models_list=models_test, dataloader=dataloader)
    t1.generate_victims(victim_idxs)
    
    # Perform Attacks Targeted
    t1.atk_params = PGD_Params()
    t1.atk_params.set_params(batch_size=batch_size, iteration = 10,
                   target = 3, x_val_min = torch.min(data_x), x_val_max = torch.max(data_x),
                   step_size = 0.01, step_norm = "inf", eps = eps, eps_norm = 2)
    
    
    
    t1.generate_advNN(adv_idx)
    t1.generate_xadv(atk_type = "pgd")
    t1.send_to_victims(victim_idxs)

    # Log Performance
    logs_adv[adv_idx]['orig_acc_transfers'] = copy.deepcopy(t1.orig_acc_transfers)
    logs_adv[adv_idx]['orig_similarities'] = copy.deepcopy(t1.orig_similarities)
    logs_adv[adv_idx]['adv_acc_transfers'] = copy.deepcopy(t1.adv_acc_transfers)
    logs_adv[adv_idx]['adv_similarities_target'] = copy.deepcopy(t1.adv_similarities)        
    logs_adv[adv_idx]['adv_target'] = copy.deepcopy(t1.adv_target_hit)

    # Miss attack Untargeted
    t1.atk_params.set_params(batch_size=batch_size, iteration = 10,
                   target = -1, x_val_min = torch.min(data_x), x_val_max = torch.max(data_x),
                   step_size = 0.01, step_norm = "inf", eps = eps, eps_norm = 2)
    t1.generate_xadv(atk_type = "pgd")
    t1.send_to_victims(victim_idxs)
    logs_adv[adv_idx]['adv_miss'] = copy.deepcopy(t1.adv_acc_transfers)
    logs_adv[adv_idx]['adv_similarities_untarget'] = copy.deepcopy(t1.adv_similarities)

# Aggregate Results Across clients 
metrics = ['orig_acc_transfers','orig_similarities','adv_acc_transfers','adv_similarities_target',
           'adv_similarities_untarget','adv_target','adv_miss'] #,'metric_alignment']

orig_acc = np.zeros([len(victim_idxs),len(victim_idxs)]) 
orig_sim = np.zeros([len(victim_idxs),len(victim_idxs)]) 
adv_acc = np.zeros([len(victim_idxs),len(victim_idxs)]) 
adv_sim_target = np.zeros([len(victim_idxs),len(victim_idxs)]) 
adv_sim_untarget = np.zeros([len(victim_idxs),len(victim_idxs)]) 
adv_target = np.zeros([len(victim_idxs),len(victim_idxs)])
adv_miss = np.zeros([len(victim_idxs),len(victim_idxs)]) 

for adv_idx in range(len(victim_idxs)):
    for victim in range(len(victim_idxs)):
        orig_acc[adv_idx,victim] = logs_adv[victim_idxs[adv_idx]][metrics[0]][victim_idxs[victim]].data.tolist()
        orig_sim[adv_idx,victim] = logs_adv[victim_idxs[adv_idx]][metrics[1]][victim_idxs[victim]].data.tolist()
        adv_acc[adv_idx,victim] = logs_adv[victim_idxs[adv_idx]][metrics[2]][victim_idxs[victim]].data.tolist()
        adv_sim_target[adv_idx,victim] = logs_adv[victim_idxs[adv_idx]][metrics[3]][victim_idxs[victim]].data.tolist()
        adv_sim_untarget[adv_idx,victim] = logs_adv[victim_idxs[adv_idx]][metrics[4]][victim_idxs[victim]].data.tolist()
        adv_target[adv_idx,victim] = logs_adv[victim_idxs[adv_idx]][metrics[5]][victim_idxs[victim]].data.tolist()
        adv_miss[adv_idx,victim] = logs_adv[victim_idxs[adv_idx]][metrics[6]][victim_idxs[victim]].data.tolist()

# Write into output file
output_file.write("orig_acc: %.6f\n" % (np.mean(orig_acc)))
output_file.write("orig_sim: %.6f\n" % (np.mean(orig_sim)))
output_file.write("adv_acc: %.6f\n" % (np.mean(adv_acc)))
output_file.write("adv_sim_target: %.6f\n" % (np.mean(adv_sim_target)))
output_file.write("adv_sim_untarget: %.6f\n" % (np.mean(adv_sim_untarget)))
output_file.write("adv_target: %.6f\n" % (np.mean(adv_target)))
output_file.write("adv_miss: %.6f\n" % (np.mean(adv_miss)))
