import numpy as np

def hidden_layer_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    layer.weight.data.uniform_(-lim, lim)
