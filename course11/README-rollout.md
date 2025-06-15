In **Reinforcement Learning (RL)**, a **rollout** refers to the process of generating a sequence of actions, states, and rewards by **running a policy** in an environment — either simulated or real.

---

### 🌀 What is a Rollout?

> A **rollout** is one complete trajectory of interaction with the environment:
> starting from an initial state, selecting actions using the agent's policy, observing resulting states and rewards, until termination or a fixed horizon.

---

### 🔁 Step-by-Step Rollout Process:

Let’s say you have:

* A policy $\pi(a|s)$
* An environment (e.g., OpenAI Gym)
* A maximum episode length or a terminal condition

**Then a rollout looks like this:**

1. **Initialize** the environment:
   $s_0 = \text{env.reset()}$

2. For $t = 0, 1, 2, ..., T$:

   * Choose action:
     $a_t \sim \pi(a_t | s_t)$
   * Apply action to the environment:
     $s_{t+1}, r_t = \text{env.step}(a_t)$
   * Record:
     $(s_t, a_t, r_t, s_{t+1})$
   * Stop if done (terminal state)

3. Return the **trajectory**:

   $$
   \tau = \left\{(s_0, a_0, r_0), (s_1, a_1, r_1), ..., (s_T, a_T, r_T)\right\}
   $$

---

### 🎯 Why Rollouts Matter:

Rollouts are used to:

* **Collect experience** for training
* Estimate **expected return** (for policy gradient methods)
* Train **value functions** or **reward models**
* Simulate agent behavior under current or updated policies

---

### 📦 In practice:

```python
obs = env.reset()
done = False
trajectory = []

while not done:
    action = policy(obs)
    next_obs, reward, done, _ = env.step(action)
    trajectory.append((obs, action, reward))
    obs = next_obs
```

---

### 🧠 Key Terms:

| Term           | Description                            |
| -------------- | -------------------------------------- |
| **Policy**     | The agent's decision-making function   |
| **Trajectory** | Sequence of states, actions, rewards   |
| **Episode**    | One complete rollout (start to end)    |
| **On-policy**  | Rollout follows the current policy     |
| **Off-policy** | Rollout uses older or different policy |

---

Let me know if you'd like to see a visual or try a rollout using `gym`.


