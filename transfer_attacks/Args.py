# Arguments used for training
class Args:
    
    def __init__(self):
        self.experiment = "cifar10"
        self.method = "FedEM"
        self.decentralized = False
        self.sampling_rate = 1.0
        self.input_dimension = None
        self.output_dimension = None
        self.n_learners= 3
        self.n_rounds = 10
        self.bz = 128
        self.local_steps = 1
        self.lr_lambda = 0
        self.lr =0.03
        self.lr_scheduler = 'multi_step'
        self.log_freq = 10
        self.device = 'cuda'
        self.optimizer = 'sgd'
        self.mu = 0
        self.communication_probability = 0.1
        self.q = 1
        self.locally_tune_clients = False
        self.seed = 1234
        self.verbose = 1
        self.save_path = 'weights/cifar/21_09_28_first_transfers/'
        self.validation = False
        
    
    def __iter__(self):
        for attr in dir(self):
            if not attr.startswith("__"):
                yield attr

                