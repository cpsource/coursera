No — **in your current code, the neural network (policy) is not trained at all**.

---

### 🧠 What's Happening Now:

The policy network is being **randomly initialized** and then used **once** to make a decision:

```python
policy = PolicyNetwork(obs_dim, 128, n_actions)  # random weights
...
action_probs = policy(obs_tensor)                # untrained output
```

Because it's untrained:

* The **output probabilities** will be somewhat random.
* The **selected action** is also effectively random.

---

### 🔧 How Training Usually Happens

To train the policy, you'd need:

* **Experience data** (rollouts): sequences of `(state, action, reward)`
* A **training objective**, like:

  * **Policy Gradient (REINFORCE)**
  * **Actor-Critic**
  * **Q-learning**
* A **loss function** and **optimizer** (e.g., `Adam`)
* A training loop that improves the policy over time.

---

### 🧪 Toy Training Example (Sketch):

```python
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)

for episode in range(1000):
    obs, _ = env.reset()
    done = False
    log_probs = []
    rewards = []

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        action_probs = policy(obs_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()

        obs, reward, done, _, = env.step(action.item())
        log_probs.append(action_dist.log_prob(action))
        rewards.append(reward)

    # Compute discounted reward
    total_reward = sum(rewards)

    # Policy gradient update
    loss = -sum(log_probs) * total_reward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

This is a minimal **REINFORCE algorithm**.

---

### ✅ Summary:

* The neural network in your code **is not trained**.
* It produces output based on **random weights**.
* To train it, you need:

  * Rewards
  * A loss function
  * Backpropagation via optimizer

Let me know if you want to add a full training loop.

