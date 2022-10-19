from cmath import exp
import torch
from torch.utils.data import DataLoader, Dataset
import scanpy as sc
from anndata import AnnData
from typing import Tuple, List
import numpy as np

class Experiment(Dataset):
    """
    Dataset for reading experiment matrices ([cell,gene] matrix)

    Parameters
    __________
    x: Tensor

    cells: ndarray

    genes: ndarray

    celltype: ndarray
    """
    def __init__(self, x, cells, genes, tissue_name, celltypes=None):
        super().__init__()
        self.x = x
        self.y = celltypes
        self.cells = cells
        self.tissue = tissue_name
        self.genes = genes


    def __getitem__(self, item):
        if isinstance(item, torch.Tensor):
            return self.x[item.item()], self.y[item.item()], self.cells[item.item()]
        else:
            return self.x[item], self.y[item], self.cells[item]
        

    def __len__(self):
        return self.x.shape[0]

    @classmethod
    def concat(cls, exp1, exp2, tissue_name = None):
        new_exp = cls(np.concatenate((exp1.x, exp2.x)), np.concatenate((exp1.cells, exp2.cells)) ,np.concatenate((exp1.genes, exp2.genes)), 
                        tissue_name, np.concatenate((exp1.y, exp2.y)) )
        return new_exp

