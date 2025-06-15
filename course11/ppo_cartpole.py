import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym

# ✅ Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Define Actor-Critic Model ---
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, hidden_dim, n_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)

# --- Hyperparameters ---
hidden_size = 64
lr = 3e-4
gamma = 0.99
eps_clip = 0.2
K_epochs = 5
T_horizon = 512

# --- Setup Environment ---
env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

policy = ActorCritic(obs_dim, hidden_size, n_actions).to(device)
optimizer = optim.Adam(policy.parameters(), lr=lr)

# --- Buffers ---
obs_buffer, action_buffer, logprob_buffer = [], [], []
reward_buffer, done_buffer, value_buffer = [], [], []

obs, _ = env.reset()
total_reward = 0

# --- Rollout Phase ---
for _ in range(T_horizon):
    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
    logits, value = policy(obs_tensor)
    dist = Categorical(logits=logits)
    action = dist.sample()

    next_obs, reward, terminated, truncated, _ = env.step(action.item())
    done = terminated or truncated

    # Save experience
    obs_buffer.append(obs_tensor)
    action_buffer.append(action)
    logprob_buffer.append(dist.log_prob(action))
    reward_buffer.append(reward)
    done_buffer.append(done)
    value_buffer.append(value)
    total_reward += reward

    obs = next_obs
    if done:
        obs, _ = env.reset()

# --- Compute returns and advantages ---
returns = []
G = 0
for r, d in zip(reversed(reward_buffer), reversed(done_buffer)):
    G = r + gamma * G * (1 - d)
    returns.insert(0, G)

returns = torch.tensor(returns, dtype=torch.float32).to(device)
value_tensor = torch.stack(value_buffer).squeeze()
advantages = (returns - value_tensor.detach()).detach()

# --- PPO Policy Update ---
obs_tensor = torch.stack(obs_buffer).to(device)
action_tensor = torch.stack(action_buffer).to(device)
logprob_tensor = torch.stack(logprob_buffer).detach().to(device)

for epoch in range(1, K_epochs + 1):
    logits, values = policy(obs_tensor)
    dist = Categorical(logits=logits)
    new_logprobs = dist.log_prob(action_tensor)
    ratio = (new_logprobs - logprob_tensor).exp()

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()
    critic_loss = (returns - values.squeeze()).pow(2).mean()
    loss = actor_loss + 0.5 * critic_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print statistics every 10 epochs
    if True: # epoch % 5 == 0 or epoch == K_epochs:
        with torch.no_grad():
            predicted_actions = dist.probs.argmax(dim=-1)
            correct = (predicted_actions == action_tensor).float().mean().item()
            print(f"Epoch {epoch:2d} | "
                  f"Actor Loss: {actor_loss.item():.4f} | "
                  f"Critic Loss: {critic_loss.item():.4f} | "
                  f"Accuracy: {correct*100:.2f}%")

print(f"\n✅ PPO training complete. Total reward during rollout: {total_reward:.2f}")

