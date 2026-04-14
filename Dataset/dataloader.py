import random
import torch
from abc import ABC
import numpy as np
from Dataset.dataset_obj import PlantDataset
from Dataset.dataset_obj import Dataset



class DataLoader:

    def __iter__(self):
        raise NotImplementedError('Child must implement this method')
    

class PlantDataLoader(DataLoader):
    """Batches the data for training and testing"""

    def __init__(self, dataset, batch_size, random= False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.random = random

    
    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.random:
            random.shuffle(indices)

        for i in range(0, len(self.dataset), self.batch_size):
            selected_indices = indices[ i: i + self.batch_size]

            img_buff = []
            y_c_buffer = []
            y_d_buffer = []


            for index in selected_indices:
                x, y_c, y_d = self.dataset[index]
                img_buff.append(x)
                y_c_buffer.append(y_c)
                y_d_buffer.append(y_d)

            yield PlantDataLoader.make_contiguous(img_buff, y_c_buffer, y_d_buffer)

    @staticmethod
    def make_contiguous(img_buff, y_cs, y_ds):
        """Returns a concatenates images and labels"""
        return torch.stack(img_buff, dim=0), torch.stack(y_cs), torch.stack(y_ds)
    

if __name__ == '__main__':
    from Dataset.dataset_obj import PlantDataset
    import os

    dataset = PlantDataset('./train-V', is_test=False)
    dataloader  =PlantDataLoader(dataset, batch_size=3, random=True)

    data_iter = iter(dataloader)
    x, y_c, y_d = next(data_iter)

    os.system('clear')
    print(f'Batched data shape: {x.shape}')
    print(f'Batched labels shape: {y_c.shape}') 
    print(f'Y lables: {y_d.shape}')

                

    