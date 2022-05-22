import random

# Write a Custom Class for Dataloader that has flexible batch size
class Custom_Dataloader:
    def __init__(self, x_data, y_data):
        self.x_data = x_data # Tensor + cuda
        self.y_data = y_data # Tensor + cuda
        
    def load_batch(self,batch_size,mode = 'test'):
        samples = random.sample(range(self.y_data.shape[0]),batch_size)
        out_x_data = self.x_data[samples].to(device='cuda')
        out_y_data = self.y_data[samples].to(device='cuda')
        
        return out_x_data, out_y_data
    
    def select_single(self):
        sample_idx = random.sample(range(self.y_data.shape[0]),1)
        x_point = self.x_data[sample_idx].to(device='cuda')
        y_point = self.y_data[sample_idx].to(device='cuda')
        
        return sample_idx, x_point, y_point