import torch.nn.functional as F

from copy import deepcopy
from utils.torch_utils import *

from transfer_attacks.Personalized_NN import *
# from transfer_attacks.Params import *
# from transfer_attacks.Transferer import *
# from transfer_attacks.Args import *
# from transfer_attacks.TA_utils import *

# from transfer_attacks.Boundary_Transferer import *
# from transfer_attacks.projected_gradient_descent import *

from transfer_attacks.Custom_Dataloader import *
from transfer_attacks.unnormalize import *
from itertools import combinations


class Client(object):
    r"""Implements one clients

    Attributes
    ----------
    learners_ensemble
    n_learners

    train_iterator

    val_iterator

    test_iterator

    train_loader

    n_train_samples

    n_test_samples

    samples_weights

    local_steps

    logger

    tune_locally:

    Methods
    ----------
    __init__
    step
    write_logs
    update_sample_weights
    update_learners_weights

    """
    def __init__(
            self,
            learners_ensemble,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            tune_locally=False,
            tune_steps=None
    ):

        self.learners_ensemble = learners_ensemble
        self.n_learners = len(self.learners_ensemble)
        self.tune_locally = tune_locally

#         if self.tune_locally:
#             self.tuned_learners_ensemble = deepcopy(self.learners_ensemble)
#         else:
#             self.tuned_learners_ensemble = None

        self.tuned_learners_ensemble = deepcopy(self.learners_ensemble)

        self.binary_classification_flag = self.learners_ensemble.is_binary_classification

        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.test_iterator = test_iterator
        self.true_train_iterator = copy.deepcopy(self.train_iterator)

        self.train_loader = iter(self.train_iterator)

        self.n_train_samples = len(self.train_iterator.dataset)
        self.n_test_samples = len(self.test_iterator.dataset)

        self.samples_weights = torch.ones(self.n_learners, self.n_train_samples) / self.n_learners

        self.local_steps = local_steps

        self.counter = 0
        self.logger = logger
        
        if tune_steps:
            self.tune_steps = tune_steps
        else:
            self.tune_steps = self.local_steps

    def get_next_batch(self):
        try:
            batch = next(self.train_loader)
        except StopIteration:
            self.train_loader = iter(self.train_iterator)
            batch = next(self.train_loader)

        return batch

    def swap_dataset_labels(self, class_count):
        y_temp = class_count - self.true_train_iterator.dataset.targets - 1
        self.train_iterator.dataset.targets = y_temp
        self.train_loader = iter(self.train_iterator)
        
        return 
    
    def reset_dataset_labels(self):
        self.train_iterator = copy.deepcopy(self.true_train_iterator)
        self.train_loader = iter(self.train_iterator)
        
        return
    
    def step(self, single_batch_flag=False, *args, **kwargs):
        """
        perform on step for the client

        :param single_batch_flag: if true, the client only uses one batch to perform the update
        :return
            clients_updates: ()
        """
        self.counter += 1
        self.update_sample_weights()
        self.update_learners_weights()

        if single_batch_flag:
            batch = self.get_next_batch()
            client_updates = \
                self.learners_ensemble.fit_batch(
                    batch=batch,
                    weights=self.samples_weights
                )
        else:
            client_updates = \
                self.learners_ensemble.fit_epochs(
                    iterator=self.train_iterator,
                    n_epochs=self.local_steps,
                    weights=self.samples_weights
                )

        # TODO: add flag arguments to use `free_gradients`
        # self.learners_ensemble.free_gradients()

        return client_updates

    def write_logs(self):
        if self.tune_locally:
            self.update_tuned_learners()

        if self.tune_locally:
            train_loss, train_acc = self.tuned_learners_ensemble.evaluate_iterator(self.val_iterator)
            test_loss, test_acc = self.tuned_learners_ensemble.evaluate_iterator(self.test_iterator)
        else:
            train_loss, train_acc = self.learners_ensemble.evaluate_iterator(self.val_iterator)
            test_loss, test_acc = self.learners_ensemble.evaluate_iterator(self.test_iterator)

        self.logger.add_scalar("Train/Loss", train_loss, self.counter)
        self.logger.add_scalar("Train/Metric", train_acc, self.counter)
        self.logger.add_scalar("Test/Loss", test_loss, self.counter)
        self.logger.add_scalar("Test/Metric", test_acc, self.counter)

        return train_loss, train_acc, test_loss, test_acc

    def update_sample_weights(self):
        pass

    def update_learners_weights(self):
        pass

    def update_tuned_learners(self):
        
#         if not self.tune_locally:
#             return
#         for learner_id, learner in enumerate(self.tuned_learners_ensemble):
#             copy_model(source=self.learners_ensemble[learner_id].model, target=learner.model)
#             learner.fit_epochs(self.train_iterator, self.tune_steps, weights=self.samples_weights[learner_id])
#             copy_model(source=learner.model, target=self.learners_ensemble[learner_id].model)

        # Forced tuning with learners ensemble
        for learner_id, learner in enumerate(self.learners_ensemble):
            learner.fit_epochs(self.train_iterator, self.tune_steps, weights=self.samples_weights[learner_id])
            


class MixtureClient(Client):
    def update_sample_weights(self):
        all_losses = self.learners_ensemble.gather_losses(self.val_iterator)
        self.samples_weights = F.softmax((torch.log(self.learners_ensemble.learners_weights) - all_losses.T), dim=1).T

    def update_learners_weights(self):
        self.learners_ensemble.learners_weights = self.samples_weights.mean(dim=1)
                


class AgnosticFLClient(Client):
    def __init__(
            self,
            learners_ensemble,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            tune_locally=False,
            tune_steps=None
    ):
        super(AgnosticFLClient, self).__init__(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            tune_steps=tune_steps
        )

        assert self.n_learners == 1, "AgnosticFLClient only supports single learner."

    def step(self, *args, **kwargs):
        self.counter += 1

        batch = self.get_next_batch()
        losses = self.learners_ensemble.compute_gradients_and_loss(batch)

        return losses


class FFLClient(Client):
    r"""
    Implements client for q-FedAvg from
     `FAIR RESOURCE ALLOCATION IN FEDERATED LEARNING`__(https://arxiv.org/pdf/1905.10497.pdf)

    """
    def __init__(
            self,
            learners_ensemble,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            q=1,
            tune_locally=False,
            tune_steps=None
    ):
        super(FFLClient, self).__init__(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            tune_steps=tune_steps
        )

        assert self.n_learners == 1, "AgnosticFLClient only supports single learner."
        self.q = q

    def step(self, lr, *args, **kwargs):

        hs = 0
        for learner in self.learners_ensemble:
            initial_state_dict = self.learners_ensemble[0].model.state_dict()
            learner.fit_epochs(iterator=self.train_iterator, n_epochs=self.local_steps)

            client_loss, _ = learner.evaluate_iterator(self.train_iterator)
            client_loss = torch.tensor(client_loss)
            client_loss += 1e-10

            # assign the difference to param.grad for each param in learner.parameters()
            differentiate_learner(
                target=learner,
                reference_state_dict=initial_state_dict,
                coeff=torch.pow(client_loss, self.q) / lr
            )

            hs = self.q * torch.pow(client_loss, self.q-1) * torch.pow(torch.linalg.norm(learner.get_grad_tensor()), 2)
            hs /= torch.pow(torch.pow(client_loss, self.q), 2)
            hs += torch.pow(client_loss, self.q) / lr

        return hs / len(self.learners_ensemble)


class Adv_MixtureClient(MixtureClient):
    """ 
    ADV client with more params -- use to PGD generate data between rounds
    """
    def __init__(
            self,
            learners_ensemble,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            tune_locally=False,
            dataset_name = 'cifar10',
            tune_steps=None
    ):
        super(Adv_MixtureClient, self).__init__(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            tune_steps=tune_steps
        )

        self.adv_proportion = 0
        self.atk_params = None
        
        # Make copy of dataset and set aside for adv training
        self.og_dataloader = deepcopy(self.train_iterator) # Update self.train_loader every iteration
        
        # Add adversarial client 
        combined_model = self.combine_learners_ensemble()
        self.altered_dataloader = self.gen_customdataloader(self.og_dataloader)
        self.adv_nn = Adv_NN(combined_model, self.altered_dataloader)
        
        self.dataset_name = dataset_name
    
    def set_adv_params(self, adv_proportion = 0, atk_params = None):
        self.adv_proportion = adv_proportion
        self.atk_params = atk_params
    
    def gen_customdataloader(self, og_dataloader):
        # Combine Validation Data across all clients as test
        data_x = []
        data_y = []

        for (x,y,idx) in og_dataloader.dataset:
            data_x.append(x)
            data_y.append(y)

        data_x = torch.stack(data_x)
        try:
            data_y = torch.stack(data_y)
        except:
            data_y = torch.tensor(data_y)
        dataloader = Custom_Dataloader(data_x, data_y)
        
        return dataloader
    
    def combine_learners_ensemble(self):

        # This is where the models are stored -- one for each mixture --> learner.model for nn
        hypotheses = self.learners_ensemble.learners

        # obtain the state dict for each of the weights 
        weights_h = []

        model_weights = self.learners_ensemble.learners_weights
        
        for h in hypotheses:
            weights_h += [h.model.state_dict()]
        
        # first make the model with empty weights
        new_model = deepcopy(hypotheses[0].model)
        new_model.eval()
        new_weight_dict = deepcopy(weights_h[0])
        for key in weights_h[0]:
            new_weight_dict[key] = model_weights[0]*weights_h[0][key]
            for i in range(1,len(model_weights)):
                new_weight_dict[key]+=model_weights[i]*weights_h[i][key]
                
        new_model.load_state_dict(new_weight_dict)
        
        return new_model
    
    def update_advnn(self):
        # reassign weights after trained
        self.adv_nn = Adv_NN(self.combine_learners_ensemble(), self.altered_dataloader)
        return
    
    def generate_adversarial_data(self):
        # Generate adversarial datapoints while recognizing idx of sampled without replacement
        
        # Draw random idx without replacement 
        num_datapoints = self.train_iterator.dataset.targets.shape[0]
        sample_size = int(np.ceil(num_datapoints * self.adv_proportion))
        sample = np.random.choice(a=num_datapoints, size=sample_size)
        x_data = self.adv_nn.dataloader.x_data[sample]
        y_data = self.adv_nn.dataloader.y_data[sample]
        
        self.adv_nn.pgd_sub(self.atk_params, x_data.cuda(), y_data.cuda())
        x_adv = self.adv_nn.x_adv
        
        return sample, x_adv
    
    def assign_advdataset(self):
        # convert dataset to normed and replace specific datapoints
        
        # Flush current used dataset with original
        self.train_iterator = deepcopy(self.og_dataloader)
        
        # adversarial datasets loop, adjust normed and push 
        sample_id, x_adv = self.generate_adversarial_data()
        
        for i in range(sample_id.shape[0]):
            idx = sample_id[i]
            x_val_normed = x_adv[i]
            try:
                x_val_unnorm = unnormalize_cifar10(x_val_normed)
            except:
                x_val_unnorm = unnormalize_femnist(x_val_normed)
        
            self.train_iterator.dataset.data[idx] = x_val_unnorm
        
        self.train_loader = iter(self.train_iterator)
        
        return
    
class Adv_MixtureClient_DVERGE(MixtureClient):
    """ 
    ADV client with more params -- use to PGD generate data between rounds
    """
    def __init__(
            self,
            learners_ensemble,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            tune_locally=False,
            dataset_name = 'cifar10',
            tune_steps=None
    ):
        super(Adv_MixtureClient_DVERGE, self).__init__(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            tune_steps=tune_steps
        )

        self.adv_proportion = 0
        self.atk_params = None
        
        # Make copy of dataset and set aside for adv training
        self.og_dataloader = deepcopy(self.train_iterator) # Update self.train_loader every iteration
        
        # Add adversarial client 
        self.altered_dataloader = self.gen_customdataloader(self.og_dataloader)
        self.num_hypotheses = len(self.learners_ensemble.learners)
        self.modes = ["indiv_h", "combination_h"]
        self.modes_idx = 1
        
        self.adv_nns = self.update_advnn()
        self.dataset_name = dataset_name
       
        
        self.train_iterator_list = []
    
    def set_adv_params(self, adv_proportion = 0, atk_params = None):
        self.adv_proportion = adv_proportion
        self.atk_params = atk_params
    
    def gen_customdataloader(self, og_dataloader):
        # Combine Validation Data across all clients as test
        data_x = []
        data_y = []

        for (x,y,idx) in og_dataloader.dataset:
            data_x.append(x)
            data_y.append(y)

        data_x = torch.stack(data_x)
        try:
            data_y = torch.stack(data_y)
        except:
            data_y = torch.tensor(data_y)
        dataloader = Custom_Dataloader(data_x, data_y)
        
        return dataloader
    
    def combine_learners_ensemble(self, hyp_idxs):

        # This is where the models are stored -- one for each mixture --> learner.model for nn
        hypotheses = self.learners_ensemble.learners

        # obtain the state dict for each of the weights 
        weights_h = []

        model_weights = self.learners_ensemble.learners_weights
        num_hyp_total = len(model_weights)
        num_hyp = len(hyp_idxs)
        scale = 1/num_hyp
        
        for h in hypotheses:
            weights_h += [h.model.state_dict()]
        
        # first make the model with empty weights
        new_model = deepcopy(hypotheses[0].model)
        new_model.eval()
        new_weight_dict = deepcopy(weights_h[0])
        for key in weights_h[0]:
            idx = hyp_idxs[0]
            new_weight_dict[key] = scale*weights_h[idx][key]
            for i in hyp_idxs[1:]:
                new_weight_dict[key]+= scale*weights_h[i][key]

        new_model.load_state_dict(new_weight_dict)
    
        return new_model
    
    def update_advnn(self):
        # reassign weights after trained
        # make X adv nn based on the number of hypotheses that are present at the models
        adv_nns = []
        
        for i in range(self.num_hypotheses):
            adv_nns += [Adv_NN(self.learners_ensemble.learners[i].model, self.altered_dataloader)]
        if self.modes_idx == 1:
            # make combinations of hypothesis
            for c in range(2,self.num_hypotheses+1):
                combo = list(combinations(range(self.num_hypotheses),c))
                for cc in combo:
                    adv_nns += [Adv_NN(self.combine_learners_ensemble(cc), self.altered_dataloader)]
                
        self.adv_nns = adv_nns
        return
    
    def generate_adversarial_data(self):
        # Generate adversarial datapoints while recognizing idx of sampled without replacement
        
        # Draw random idx without replacement 
        num_datapoints = self.train_iterator.dataset.targets.shape[0]
        sample_size = int(np.ceil(num_datapoints * self.adv_proportion))
        sample = np.random.choice(a=num_datapoints, size=sample_size)
        
        # Sample the proportion and split the sampled dataset into N equal groups per hypotheses
        sample_groups = {}
        x_adv_groups = {} # Splitting by hypotheses
        
        num_nns = len(self.adv_nns)
        
        for i in range(num_nns):
            sub_sample = sample[int(np.floor(i*sample.shape[0]/num_nns)):
                                int(np.floor((i+1)*sample.shape[0]/num_nns))]
            sample_groups[i] = sub_sample
            x_data = self.altered_dataloader.x_data[sub_sample]
            y_data = self.altered_dataloader.y_data[sub_sample]
            
            self.adv_nns[i].pgd_sub(self.atk_params, x_data.cuda(), y_data.cuda())
            x_adv_groups[i] = self.adv_nns[i].x_adv
        
        return sample_groups, x_adv_groups
    
    def assign_advdataset(self):
        # convert dataset to normed and replace specific datapoints
        
        # Flush current used dataset with original
        self.train_iterator = deepcopy(self.og_dataloader)
        train_iterator_list = []
        
        num_nns = len(self.adv_nns)
        
        # adversarial datasets loop, adjust normed and push 
        sample_id_groups, x_adv_groups = self.generate_adversarial_data()
        
        for j in range(self.num_hypotheses):
            train_iterator_list += [deepcopy(self.og_dataloader)]
            
            for k in range(num_nns):
#                 if j != k:
                # To revert to bad DVERGE tab from here
                sample_id = sample_id_groups[k]
                x_adv = x_adv_groups[k]

                for i in range(sample_id.shape[0]):
                    idx = sample_id[i]
                    x_val_normed = x_adv[i]
                    if self.dataset_name == 'cifar10' or self.dataset_name == 'cifar100':
                        x_val_unnorm = unnormalize_cifar10(x_val_normed)
                    elif self.dataset_name == 'mnist' or self.dataset_name == 'femnist':
                        x_val_unnorm = unnormalize_femnist(x_val_normed)
                    else:
                        print("Error: Dataset not recognized")

                    train_iterator_list[j].dataset.data[idx] = x_val_unnorm
                    # To here and uncomment j!=k
        
        self.train_iterator_list = train_iterator_list
        self.train_loader = iter(self.train_iterator)
        
        return
    
    def step(self, single_batch_flag=False, *args, **kwargs):
        """
        perform on step for the client

        :param single_batch_flag: if true, the client only uses one batch to perform the update
        :return
            clients_updates: ()
        """
        self.counter += 1
        self.update_sample_weights()
        self.update_learners_weights()

        if single_batch_flag:
            batch = self.get_next_batch()
            client_updates = \
                self.learners_ensemble.fit_batch(
                    batch=batch,
                    weights=self.samples_weights
                )
        else:
            if len(self.train_iterator_list) == 0:
                client_updates = \
                    self.learners_ensemble.fit_epochs(
                        iterator=self.train_iterator,
                        n_epochs=self.local_steps,
                        weights=self.samples_weights)
            else:
                client_updates = \
                    self.learners_ensemble.fit_epochs_multiple_iterators(
                        iterators=self.train_iterator_list,
                        n_epochs=self.local_steps,
                        weights=self.samples_weights)
                
        return client_updates
    
class Adv_Client(Client):
    """ 
    ADV client with more params -- use to PGD generate data between rounds
    """
    def __init__(
            self,
            learners_ensemble,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            tune_locally=False,
            dataset_name = 'cifar10'
    ):
        super(Adv_Client, self).__init__(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally
        )

        self.adv_proportion = 0
        self.atk_params = None
        
        # Make copy of dataset and set aside for adv training
        self.og_dataloader = deepcopy(self.train_iterator) # Update self.train_loader every iteration
        
        # Add adversarial client 
        combined_model = self.combine_learners_ensemble()
        self.altered_dataloader = self.gen_customdataloader(self.og_dataloader)
        self.adv_nn = Adv_NN(combined_model, self.altered_dataloader)
        self.dataset_name = dataset_name
    
    def set_adv_params(self, adv_proportion = 0, atk_params = None):
        self.adv_proportion = adv_proportion
        self.atk_params = atk_params
    
    def gen_customdataloader(self, og_dataloader):
        # Combine Validation Data across all clients as test
        data_x = []
        data_y = []

        for (x,y,idx) in og_dataloader.dataset:
            data_x.append(x)
            data_y.append(y)

        data_x = torch.stack(data_x)
        try:
            data_y = torch.stack(data_y)
        except:
            data_y = torch.tensor(data_y)
        dataloader = Custom_Dataloader(data_x, data_y)
        
        return dataloader
    
    def combine_learners_ensemble(self):

        # This is where the models are stored -- one for each mixture --> learner.model for nn
        hypotheses = self.learners_ensemble.learners

        # obtain the state dict for each of the weights 
        weights_h = []

        model_weights = self.learners_ensemble.learners_weights
        
        for h in hypotheses:
            weights_h += [h.model.state_dict()]
        
        # first make the model with empty weights
        new_model = deepcopy(hypotheses[0].model)
        new_model.eval()
        new_weight_dict = deepcopy(weights_h[0])
        for key in weights_h[0]:
            htemp = model_weights[0]*weights_h[0][key]
            for i in range(1,len(model_weights)):
                htemp+=model_weights[i]*weights_h[i][key]
            new_weight_dict[key] = htemp
        new_model.load_state_dict(new_weight_dict)
        
        return new_model
    
    def update_advnn(self):
        # reassign weights after trained
        self.adv_nn = Adv_NN(self.combine_learners_ensemble(), self.altered_dataloader)
        return
    
    def generate_adversarial_data(self):
        # Generate adversarial datapoints while recognizing idx of sampled without replacement
        
        # Draw random idx without replacement 
        num_datapoints = self.train_iterator.dataset.targets.shape[0]
        sample_size = int(np.ceil(num_datapoints * self.adv_proportion))
        sample = np.random.choice(a=num_datapoints, size=sample_size)
        x_data = self.adv_nn.dataloader.x_data[sample]
        y_data = self.adv_nn.dataloader.y_data[sample]
        
        self.adv_nn.pgd_sub(self.atk_params, x_data.cuda(), y_data.cuda())
        x_adv = self.adv_nn.x_adv
        
        return sample, x_adv
    
    def assign_advdataset(self):
        # convert dataset to normed and replace specific datapoints
        
        # Flush current used dataset with original
        self.train_iterator = deepcopy(self.og_dataloader)
        
        # adversarial datasets loop, adjust normed and push 
        sample_id, x_adv = self.generate_adversarial_data()
        
        for i in range(sample_id.shape[0]):
            idx = sample_id[i]
            x_val_normed = x_adv[i]
            try:
                x_val_unnorm = unnormalize_cifar10(x_val_normed)
            except:
                x_val_unnorm = unnormalize_femnist(x_val_normed)
        
            self.train_iterator.dataset.data[idx] = x_val_unnorm
        
        self.train_loader = iter(self.train_iterator)
        
        return