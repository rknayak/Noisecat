
import torch
import numpy as np
import torch.nn as nn
from torch.nn import Conv2d,ReLU,BatchNorm2d,Linear

class Net(nn.Module):
    def __init__(self,input_size=1,layer_sizes=[20]*23,filter_size=3,padding=1):
        super().__init__()
        
        self.conv_layers = nn.ModuleList()
        self.relu_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        for hidden_i in range(len(layer_sizes)):
            self.conv_layers.append(nn.Conv2d(input_size if hidden_i==0 else layer_sizes[hidden_i-1],
                                              layer_sizes[hidden_i],filter_size,padding=padding,bias=False))
            self.bn_layers.append(nn.BatchNorm2d(layer_sizes[hidden_i]))
            self.relu_layers.append(ReLU(inplace=True))

        self.conv_layer = nn.Conv2d(layer_sizes[-1],input_size,filter_size,padding=padding,bias=False)  # last layer
        
        
    def forward(self,x):
        
        for hidden_i, layer in enumerate( self.conv_layers ):
            out = layer(x if hidden_i==0 else out)
            out = self.bn_layers[hidden_i](out)
            out = self.relu_layers[hidden_i](out)
            
        out = self.conv_layer(out) 
        
        out = out+x
        
        return out

