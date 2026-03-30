import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split

import os 

from box import Box
import yaml

from .dataset import SonarSimDataset

class SonarSimDataModule(pl.LightningDataModule):

    def __init__(self, root_dir, batch_size, num_workers, transforms, frames_in_series):
        
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size 
        self.transforms = transforms
        self.window_size = frames_in_series
        self.num_workers = num_workers

    def setup(self, stage):
        if stage == "fit" or stage is None:
            train_pth = os.path.join(self.root_dir, 'train')
            self.train_dataset = SonarSimDataset(train_pth, self.window_size, transform=self.transforms, revert_sequence_p = 0.5)
            
            val_pth = os.path.join(self.root_dir, 'val')
            self.val_dataset = SonarSimDataset(val_pth, self.window_size, transform=self.transforms, revert_sequence_p = 0.0)
        if stage == "test" or stage is None:
            test_pth = os.path.join(self.root_dir, 'test')
            self.test_dataset = SonarSimDataset(test_pth, self.window_size, transform=self.transforms, revert_sequence_p = 0.0)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.num_workers)


















# class SonarDataModule(pl.LightningDataModule):
#     def __init__(self, data_dir: str, batch_size: int = 32):
#         super().__init__()
#         self.data_dir = data_dir
#         self.batch_size = batch_size

#     def setup(self, stage: str = None):
#         # Ta metoda jest wywoływana na każdym GPU/maszynie.
#         # Tutaj ładujesz swój Dataset i dzielisz na train/val/test
        
#         pelny_dataset = TwojSuperDataset(self.data_dir)
        
#         # Przykładowy podział 80/20
#         train_size = int(0.8 * len(pelny_dataset))
#         val_size = len(pelny_dataset) - train_size
        
#         self.train_dataset, self.val_dataset = random_split(
#             pelny_dataset, [train_size, val_size]
#         )

#     def train_dataloader(self):
#         # Zwracasz po prostu standardowy Dataloader PyTorcha
#         return DataLoader(
#             self.train_dataset, 
#             batch_size=self.batch_size, 
#             shuffle=True, 
#             num_workers=4 # Przyspiesza ładowanie danych
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.val_dataset, 
#             batch_size=self.batch_size, 
#             shuffle=False, 
#             num_workers=4
#         )
