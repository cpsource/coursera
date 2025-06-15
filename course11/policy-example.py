import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# --- Define Policy Network ---
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)

# --- Create environment ---
env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

# --- Initialize policy and optimizer ---
policy = PolicyNetwork(obs_dim, 128, n_actions)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
gamma = 0.99  # Discount factor

# --- Training loop ---
for episode in range(500):
    obs, _ = env.reset()
    log_probs = []
    rewards = []
    done = False

    # --- Collect rollout ---
    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        action_probs = policy(obs_tensor)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()

        log_probs.append(dist.log_prob(action))

        obs, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        done = terminated or truncated

    # --- Compute discounted rewards ---
    discounted_rewards = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted_rewards.insert(0, R)

    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)  # Normalize

    # --- Compute loss and update policy ---
    loss = -torch.sum(torch.stack(log_probs) * discounted_rewards)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if episode % 25 == 0:
        print(f"Episode {episode}, Total Reward: {sum(rewards):.1f}")

env.close()

