obs = env.reset()
done = False
trajectory = []

while not done:
    action = policy(obs)
    next_obs, reward, done, _ = env.step(action)
    trajectory.append((obs, action, reward))
    obs = next_obs


