import itertools
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import copy

# Import Relevant Libraries
from transfer_attacks.Transferer import *

class DA_Transferer(Transferer): 
    """
    - Load all the datasets but separate them
    - Intermediate values of featues after 2 convolution layers
    """
    
    def __init__(self, filename:str, config_name = None):
        super(DA_Transferer, self).__init__(filename=filename, config_name=config_name)
        
        # Hold Onto the data
        self.DA_x = {} # Indexed by client id of dataset
        self.DA_y = {} # Also can be indexed by class - double dictionary for class-client pair
        self.DA_intermed = {}
        self.loader_i = {}
        
        # Split ADV and ROBUST DAta
        self.adv_DA_x = {}
        self.adv_DA_y = {}
        self.robust_DA_x = {}
        self.robust_DA_y = {}
        self.adv_DA_intermed = {}
        self.robust_DA_intermed = {}
        
        # Data division information
        self.mode = None
        self.client_idxs = None
        self.classes = None
        
        # Gaussian Extraction
        self.gaussian_ustd= {}
        self.adv_gaussian_ustd = {}
        self.robust_gaussian_ustd = {}
        
        # PCA Extraction
        self.PCA_data= {}
        self.dimension = None
        self.explained_var_ratio = None
        
        
    def load_niid_data(self, clients = [0,1,2,3,4,5,6,7]):
        """
        Store all data in dictionary (pre-load) separated by client idx
        """
        
        self.client_idxs = clients
        
        # Import Data Loader for this FL set
        file_indices = [i for i in range(self.config['num_sets'])]
        client_slice = len(file_indices)//self.config['num_clients']
        
        for client_idx in clients:
            if client_idx != self.advNN_idx:
                self.loader_i[client_idx] = Dataloader(file_indices,[client_idx*(client_slice),min((client_idx+1)*(client_slice),35)])  
                self.loader_i[client_idx].load_training_dataset()
                self.loader_i[client_idx].load_testing_dataset()
            else:
                self.loader_i[client_idx] = self.advNN.dataloader
        
         
    def set_data(self, mode='client', datasets = range(8), batch_size = 50, classes = [0,1]):
        """
        - fill DA_x, DA_y with relevant data according to dictionary
        modes:
            - 'client' - load all data for specified clients without class filtering
            - 'class'  - load all data and filters by class for different classes separately for single client
            - 'both'   - load all data and filters by class for different classes separately for multiple
        datasets:
            which clients to take dataset from
        """
        
        self.mode = mode
        self.client_idxs = datasets
        self.classes = classes
        self.DA_x = {} # Reset
        self.DA_y = {}

        
        # store data differently based on what the desired mode is
        if mode == 'client':
            for i in datasets:
                image_data = self.loader_i[i].load_batch(batch_size, mode='test')
                self.DA_x[i] = torch.Tensor(image_data['input']).reshape(batch_size,1,28,28) # Eliminate magic number
                self.DA_y[i] = torch.Tensor(image_data['label']).type(torch.LongTensor)
            
        elif mode == 'class':
            idx = datasets[0] # If given multiple classes take the first one
            loader = self.loader_i[idx]
            y = np.array(loader.test_dataset['user_data']['y'])
            for c in classes:
                args = np.argwhere(y==c)
                np.random.shuffle(args)
                
                # If not enough samples
                if args.shape[0] < batch_size:
                    batch_size_temp = args.shape[0]
                else: 
                    batch_size_temp = batch_size
                
                args = args[0:batch_size_temp]
                args = args.ravel()
                
                # Append data point one by one
                self.DA_x[c] = torch.Tensor(np.array(loader.test_dataset['user_data']['x'])[args]).reshape(batch_size_temp,1,28,28)
                self.DA_y[c] = torch.Tensor(np.array(loader.test_dataset['user_data']['y'])[args])
        
        elif mode == 'both':
            for i in datasets:
                loader = self.loader_i[i]
                y = np.array(loader.test_dataset['user_data']['y'])
                
                self.DA_x[i] = {}
                self.DA_y[i] = {}
                
                for c in classes:
                    args = np.argwhere(y==c)
                
                    # If not enough samples
                    if args.shape[0] < batch_size:
                        batch_size_temp = args.shape[0]
                    else: 
                        batch_size_temp = batch_size

                    args = args[0:batch_size_temp]
                    args = args.ravel()
                    self.DA_x[i][c] = torch.Tensor(np.array(loader.test_dataset['user_data']['x'])[args]).reshape(batch_size_temp,1,28,28)
                    self.DA_y[i][c] = torch.Tensor(np.array(loader.test_dataset['user_data']['y'])[args])
                    
        else:
            raise Exception("Invalid data analysis mode") 
        
    def set_data_advNN(self):
        """
        Extract data from ADV NN that were used to fool other neural networks
        Make sure the mode is set to "client" no matter what
        """

        # RESET
        self.adv_DA_x = {}
        self.adv_DA_y = {}
        self.robust_DA_x = {}
        self.robust_DA_y = {}
        
        for idx in self.victim_idxs:
            self.adv_DA_x[idx] = self.x_orig[self.adv_indices[idx]]
            self.adv_DA_y[idx] = self.y_orig[self.adv_indices[idx]]
            self.robust_DA_x[idx] = self.x_orig[self.robust_indices[idx]]
            self.robust_DA_y[idx] = self.y_orig[self.robust_indices[idx]]

    
    def forward_neck(self, x):
        """
        Only forward through neck to get upto intermediate flattened layer
        """
    
        if torch.cuda.is_available():
                x = x.cuda()
        
        x = self.advNN.neck.forward(x)
        
        return x
    
    def forward_pass(self):
        
        # Turn off dropout 
        self.advNN.eval()
        
        self.DA_intermed = {}
        self.adv_DA_intermed = {}
        self.robust_DA_intermed = {}

        if self.mode == 'client' or self.mode == 'class':
            for client_idx, value in self.DA_x.items():
                self.DA_intermed[client_idx] = self.forward_neck(value)
                
            for client_idx, value in self.adv_DA_x.items():
                try:
                    self.adv_DA_intermed[client_idx] = self.forward_neck(value)
                except:
                    pass
            
            for client_idx, value in self.robust_DA_x.items():
                try:
                    self.robust_DA_intermed[client_idx] = self.forward_neck(value)
                except:
                    pass

        elif self.mode == 'both':
            for client_idx, classes in self.DA_x.items():
                self.DA_intermed[client_idx] = {}
                for class_idx, value in classes.items():
                    self.DA_intermed[client_idx][class_idx] = self.forward_neck(value)
    
    
    def obtain_gaussian(self):
        """
        Mean and standard deviation log of every data set split
        """
        
        # Reset Dictionary
        self.gaussian_ustd= {}
        self.adv_gaussian_ustd = {}
        self.robust_gaussian_ustd = {}
        
        if self.mode == 'client':
            self.gaussian_ustd['info'] = ("mean","std","client")
            for client_idx in self.client_idxs:
                self.gaussian_ustd[client_idx] = {}
                data = self.DA_intermed[client_idx]
                self.gaussian_ustd[client_idx]['mean'] = torch.mean(data,0)
                self.gaussian_ustd[client_idx]['std'] = torch.std(data,0)
                
            for client_idx in self.adv_DA_x.keys():
                self.adv_gaussian_ustd[client_idx] = {}
                data = self.adv_DA_intermed[client_idx]
                self.adv_gaussian_ustd[client_idx]['mean'] = torch.mean(data,0)
                self.adv_gaussian_ustd[client_idx]['std'] = torch.std(data,0)
                
            for client_idx in self.robust_DA_x.keys():
                self.robust_gaussian_ustd[client_idx] = {}
                data = self.robust_DA_intermed[client_idx]
                self.robust_gaussian_ustd[client_idx]['mean'] = torch.mean(data,0)
                self.robust_gaussian_ustd[client_idx]['std'] = torch.std(data,0)
                
        elif self.mode == 'class':
            self.gaussian_ustd['info'] = ("mean","std","class")
            for class_idx in self.classes:
                self.gaussian_ustd[class_idx] = {}
                data = transferer.DA_intermed[class_idx]
                self.gaussian_ustd[class_idx]['mean'] = torch.mean(data,0)
                self.gaussian_ustd[class_idx]['std'] = torch.std(data,0)
                
        elif self.mode == 'both':
            self.gaussian_ustd['info'] = ("mean","std","both")
            for client_idx in self.client_idxs:
                self.gaussian_ustd[client_idx] = {}
                for class_idx in self.classes:
                    self.gaussian_ustd[client_idx][class_idx] = {}
                    data = transferer.DA_intermed[client_idx][class_idx]
                    self.gaussian_ustd[client_idx][class_idx]['mean'] = torch.mean(data,0)
                    self.gaussian_ustd[client_idx][class_idx]['std'] = torch.std(data,0)
    
    def obtain_PCA(self, analyze_atk = False, advrobust_idx = 1, dim=2):
        """
        Dimension reduction per data point from 400 --> 2 for all data points
        https://github.com/mGalarnyk/Python_Tutorials/blob/master/Sklearn/PCA/PCA_Data_Visualization_Iris_Dataset_Blog.ipynb
        """
        # Reset Dictionary
        self.PCA_data = {}
        self.dimension = dim
        
        output_dim = 400 # Eliminate magic numbers
        data = np.empty([0,output_dim])
        
        # Convert data to numpy and gather them together 
        # Keep index of each class pair in dictionary
        if self.mode == 'client':
            indices = np.empty((0,2)) # key(0=orig data, 1 = adv_data, 2 = robust_data)
            for client_idx in self.client_idxs:
                new_data = self.DA_intermed[client_idx].cpu().detach().numpy()
                data = np.append(data, new_data, axis=0)
                index = np.ones((new_data.shape[0],1)) * client_idx
                index = np.append(index,(np.zeros((new_data.shape[0],1))), axis=1)
                indices = np.append(indices, index,axis=0)
                
            if analyze_atk == True:
                new_data = self.adv_DA_intermed[advrobust_idx].cpu().detach().numpy()
                data = np.append(data, new_data, axis=0)
                index = np.ones((new_data.shape[0],1)) * advrobust_idx
                index = np.append(index,(np.ones((new_data.shape[0],1))), axis=1)
                indices = np.append(indices, index,axis=0)
                
                new_data = self.robust_DA_intermed[advrobust_idx].cpu().detach().numpy()
                data = np.append(data, new_data, axis=0)
                index = np.ones((new_data.shape[0],1)) * advrobust_idx
                index = np.append(index,(np.ones((new_data.shape[0],1))*2), axis=1)
                indices = np.append(indices, index,axis=0)
                
        elif self.mode == 'class':
            indices = np.empty(0)
            for class_idx in self.classes:
                new_data = self.DA_intermed[class_idx].cpu().detach().numpy()
                data = np.append(data, new_data,axis=0)
                index = np.ones(new_data.shape[0]) * class_idx
                indices = np.append(indices, index,axis=0)
                
        elif self.mode == 'both':
            indices = np.empty((0,2))
            for client_idx in self.client_idxs:
                for class_idx in self.classes:
                    new_data = self.DA_intermed[client_idx][class_idx].cpu().detach().numpy()
                    data = np.append(data,new_data,axis=0)
                    index = np.append(np.ones([new_data.shape[0],1])*client_idx, np.ones([new_data.shape[0],1])*class_idx,axis=1)
                    indices = np.append(indices,index,axis=0)
        
        
        # Standardize across dimensions of data
        data = StandardScaler().fit_transform(data)
        
        # Run PCA on data with dimension 
        pca = PCA(n_components=dim)
        principalComponents = pca.fit_transform(data)
        self.explained_var_ratio = pca.explained_variance_ratio_
        
        self.PCA_data['data'] = principalComponents
        self.PCA_data['labels'] = indices
