import torch

# Create a Bernoulli distribution with p=0.7
bernoulli = torch.distributions.Bernoulli(probs=0.7)

# Sample from it
sample = bernoulli.sample()
print(sample.item())  # Either 0.0 or 1.0

samples = bernoulli.sample((10,))
print(samples)

#samples = bernoulli.sample((10,))
streak = 0
max_streak = 0
for s in samples:
    if s == 1:
        streak += 1
        max_streak = max(max_streak, streak)
    else:
        streak = 0

print(f"max_streak = {max_streak}")

