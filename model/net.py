"""
Neural Network Architecture

Author: Arash Khoeini
Email: akhoeini@sfu.ca
"""

import torch.nn as NN
#from model.dsbn import DomainSpecificBatchNorm1d

def full_block(input_dim, output_dim,p_drop):
    return NN.Sequential(
        NN.Linear(input_dim, output_dim, bias=True),
        NN.LayerNorm(output_dim),
        NN.ELU(),
        NN.Dropout(p=p_drop),
    )


class Net(NN.Module):

    def __init__(self, input_dim, z_dim, h_dim, p_drop):
        super(Net, self).__init__()
      
        self.encoder = NN.Sequential()
        self.encoder.append(full_block(input_dim, h_dim[0],  0))
        for i in range(1, len(h_dim)):
            self.encoder.append(full_block(h_dim[i-1], h_dim[i],  p_drop))
        self.encoder.append(full_block(h_dim[-1], z_dim,  p_drop))
        
        self.projection = NN.Sequential(full_block(z_dim, z_dim, 0), full_block(z_dim, z_dim, 0)) 

    def forward(self, X):

        z = self.encoder(X)
        return self.decoder(z)