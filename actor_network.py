import torch.nn as nn
import torch.nn.functional as functional
import network

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.output_layer = nn.Linear(300, action_dim)

        network.hidden_layer_init(self.fc1)
        network.hidden_layer_init(self.fc2)
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states):
        x = functional.relu(self.fc1(states))
        x = functional.relu(self.fc2(x))
        return functional.tanh(self.output_layer(x))
