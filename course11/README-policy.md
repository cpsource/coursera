In **neural networks for reinforcement learning**, a **policy** is a model (usually a neural network) that maps **states (observations)** to **actions** — either:

* **Deterministically**: always picks the best action
* **Stochastically**: returns a probability distribution over actions

---

### 🧠 Intuition:

> A **policy** is the agent’s brain. It decides **what to do next**, given the current state.

---

### 📌 Formal Definition:

A policy is denoted:

* $\pi(a | s)$ — the probability of taking action $a$ given state $s$

---

## ✅ Python Example: Policy with a Neural Network (PyTorch)

We’ll build a simple **policy network** for a toy discrete action space like in `CartPole` (OpenAI Gym):

```python
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
```

---

### 🧪 Example Usage:

```python
import gym

# Create environment and policy
env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]   # e.g., 4
n_actions = env.action_space.n             # e.g., 2

policy = PolicyNetwork(obs_dim, 128, n_actions)

# Simulate a rollout
obs = env.reset()
obs_tensor = torch.tensor(obs, dtype=torch.float32)

# Get action probabilities from the policy
action_probs = policy(obs_tensor)
action = torch.multinomial(action_probs, num_samples=1).item()  # sample an action

print("Action probabilities:", action_probs.detach().numpy())
print("Chosen action:", action)
```

---

### ✅ Summary:

| Concept               | Meaning                                       |
| --------------------- | --------------------------------------------- |
| **Policy**            | NN that chooses actions based on state        |
| **Stochastic policy** | Outputs probability distribution over actions |
| **Used for**          | RL tasks like CartPole, Atari, MuJoCo, etc.   |

Let me know if you want to implement this using a continuous action space or DQN-style deterministic policy.


