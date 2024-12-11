import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


"""
Neural Network model structure for the DQN agent
"""
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(3, 128)
        self.linear2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 2)

    def forward(self, x):
        y = self.output(F.relu(self.linear2((F.relu(self.linear1(x))))))
        return y


"""
DQN agent for reinforcement learning
"""
class Loaded_agent:
    def __init__(self, device, trained_model_path=None):
        # Set up model
        self.device = device
        self.model = NeuralNetwork().to(device)

        if trained_model_path:
            self.load_model(trained_model_path)

        # Set up experience buffer (Not needed for evaluation, but kept here for structure)
        self.buffer = [[], [], [], []]
        self.buffer_size = 10000

        # Training hyperparameters (these are not used if loading a trained model)
        self.lr = 0.001
        self.batch_size = 64
        self.gamma = 0.2
        self.epsilon = 0.0

        # Set up training helpers (Not used for evaluation)
        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def act(self, state, train=False):
        state = torch.Tensor(state).to(self.device)

        # Choose action with max q value or select randomly (if in training mode and epsilon > random value)
        if np.random.rand() > self.epsilon or not train:
            with torch.no_grad():
                return self.model(state).data.max(0)[1].detach().cpu()  # Choose action with max q value
        else:
            return torch.randint(2, (1,))[0]  # Random action

    def load_model(self, model_path):
        """ Load the trained model from a file """
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()  # Set the model to evaluation mode

    def save_model(self, model_path):
        """ Save the current model to a file """
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, model_path)
