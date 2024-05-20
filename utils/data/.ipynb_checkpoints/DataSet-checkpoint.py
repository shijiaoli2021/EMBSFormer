import torch
import argparse
import configparser
import numpy as np
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
import utils.data.functions as functions


class SpatioTemtoralSeqDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_path,
                 tem_speed_path,
                 adj_path,
                 minute_offset,
                 dataset_name,
                 batch_size,
                 config_path,
                 time_len=None,
                 split_ratio=0.7,
                 val_ratio=0.1,
                 normalize=True,
                 ):
        super(SpatioTemtoralSeqDataModule, self).__init__()
        self._data_path = data_path
        self._tem_speed_path = tem_speed_path
        self._dataset_name = dataset_name
        self._adj_path = adj_path
        self.stats = None
        self.time_len = time_len
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.normalize = normalize
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.val_ratio = val_ratio
        if minute_offset == None:
            self.minute_offset = 0
        else:
            self.minute_offset = minute_offset
        self._data = functions.dataPre(data_path,
                                           tem_speed_path,
                                            self.minute_offset,
                                           self.config.getboolean('Data','add_time_in_day'),
                                           self.config.getboolean('Data','add_day_in_week'),
                                           self.config.getboolean('Data', 'add_tem_speed'),
                                           self.config.getboolean('Data', 'add_holiday'),
                                           self.config.getboolean('Data', 'clean_junk_data'),
                                           dataset_name)
        if "adj_mx.pkl" in adj_path:
            self._adj = functions.get_adj_matrix_METR(adj_path, int(self.config['Data']['nodes_num']))
        else:
            self._adj = functions.get_adj_matrix_PEM(adj_path, int(self.config['Data']['nodes_num']))
        self._laplacian = torch.FloatTensor(functions.get_laplace_matrix(self._adj, functions.get_d_matrix(self._adj)))

    # loading train_dataset and val_dataset
    def setup(self, stage: str = None):
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            self.stats
        ) = functions.generate_torch_datasets2(
            self._data,
            self.config,
            split_ratio=self.split_ratio,
            val_ratio=self.val_ratio,
            normalize=self.normalize,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


