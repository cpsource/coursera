import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# A simple policy network: maps 1D input → logits for 2 actions
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 2)

    def forward(self, x):
        return self.fc(x)

# Initialize
policy = PolicyNet()
optimizer = optim.Adam(policy.parameters(), lr=0.01)

# Training loop
for episode in range(200):
    x = torch.tensor([[1.0]]) if torch.rand(1).item() > 0.5 else torch.tensor([[-1.0]])

    # Forward pass
    logits = policy(x)
    dist = Categorical(logits=logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)

    # Simulated reward
    # If x > 0 → reward 1 for action 1
    # If x < 0 → reward 1 for action 0
    correct = int((x.item() > 0 and action.item() == 1) or (x.item() < 0 and action.item() == 0))
    reward = torch.tensor(float(correct))

    # Log-derivative trick:
    loss = -log_prob * reward

    # Update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if episode % 20 == 0:
        print(f"Episode {episode}, x={x.item():.1f}, action={action.item()}, reward={reward.item()}, loss={loss.item():.4f}")

