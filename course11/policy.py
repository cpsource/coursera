import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple feedforward policy for discrete actions (e.g., CartPole)
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        action_probs = F.softmax(x, dim=-1)  # stochastic policy
        return action_probs

import gym

# Create environment and policy
env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]   # e.g., 4
n_actions = env.action_space.n             # e.g., 2

policy = PolicyNetwork(obs_dim, 128, n_actions)

# Simulate a rollout
obs,_ = env.reset()
obs_tensor = torch.tensor(obs, dtype=torch.float32)

# Get action probabilities from the policy
action_probs = policy(obs_tensor)
action = torch.multinomial(action_probs, num_samples=1).item()  # sample an action

print("Action probabilities:", action_probs.detach().numpy())
print("Chosen action:", action)


