import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f


class NormalizationLayer(nn.Module):
    def __init__(self, mean, std):
        super(NormalizationLayer, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        return (x - self.mean) / self.std


class Actor(nn.Module):
    def __init__(self, dim_state, dim_action, mean, std, env):
        super(Actor, self).__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.mean = mean
        self.std = std
        self.env = env
        self.norm_layer = NormalizationLayer(self.mean, self.std)
        self.fc1 = nn.Linear(self.dim_state, 32)
        self.fc2 = nn.Linear(32, 32)
        self.center_direction = nn.Linear(32, 1)
        self.center_distance = nn.Linear(32, 1)
        self.radius = nn.Linear(32, 1)

    def forward(self, x):
        x = self.norm_layer(x)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        # Action about direction of the center
        center_x = (torch.tanh(self.center_direction(x)) *
                    (self.env.range_center_x[1] - self.env.range_center_x[0]) / 2
                    + (self.env.range_center_x[1] + self.env.range_center_x[0]) / 2)
        # Action about distance of the center
        center_y = (torch.tanh(self.center_direction(x)) *
                    (self.env.range_center_y[1] - self.env.range_center_y[0]) / 2
                    + (self.env.range_center_y[1] + self.env.range_center_y[0]) / 2)
        # Action about the radius
        radius = torch.tanh(self.radius(x))
        radius = radius * 150 + 350
        return torch.cat((center_x, center_y, radius), -1)


class Critic(nn.Module):
    def __init__(self, dim_state, dim_action, mean, std):
        super(Critic, self).__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.mean = mean
        self.std = std
        self.norm_layer = NormalizationLayer(self.mean, self.std)
        self.fc1 = nn.Linear(self.dim_state + self.dim_action, 32)
        self.fc2 = nn.Linear(32, 32)
        self.q_out = nn.Linear(32, 3)  # 修改输出层为3个值

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = self.norm_layer(x)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        q_values = self.q_out(x)  # 输出3个值
        return q_values

