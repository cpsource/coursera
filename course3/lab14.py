# Lab: Building a Deep Q-Network with Keras

import gym
import numpy as np

# Create the environment 
env = gym.make('CartPole-v1')

# Set random seed for reproducibility 
np.random.seed(42)
env.reset(seed=42)

# Step 2: Define the DQN model

# Suppress warnings for a cleaner notebook or console experience
import warnings
warnings.filterwarnings('ignore')

# Disable warnings for a cleaner notebook or console experience
def warn(*args, **kwargs):
    pass
warnings.warn = warn

# Import necessary libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def build_model(state_size, action_size):
    model = Sequential()
    model.add(Dense(24, input_dim=state_size, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
model = build_model(state_size, action_size)

# Step 3: Implement the replay buffer

from collections import deque
import random

memory = deque(maxlen=2000)
def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

# Step 4: Implement the epsilon-greedy policy

epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

def act(state):
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)
    q_values = model.predict(state)
    return np.argmax(q_values[0])

# Step 5: Implement the Q-learning update¶

def replay(batch_size):
    global epsilon
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target = reward + gamma * np.amax(model.predict(next_state)[0])
        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# Step 6: Train the DQN

# Training loop
episodes = 50  # More episodes to ensure sufficient training
batch_size = 32  # Mini-batch size for replay training
gamma = 0.95  # Discount factor for future rewards

for e in range(episodes):
    state = env.reset()
    if isinstance(state, tuple):  # Handle tuple output
        state = state[0]
    state = np.reshape(state, [1, state_size])

    for time in range(200):  # Max steps per episode
        # Choose action using epsilon-greedy policy
        action = act(state)

        # Perform action in the environment
        result = env.step(action)
        if len(result) == 4:  # Handle 4-value output
            next_state, reward, done, _ = result
        else:  # Handle 5-value output
            next_state, reward, done, _, _ = result

        if isinstance(next_state, tuple):  # Handle tuple next_state
            next_state = next_state[0]
        next_state = np.reshape(next_state, [1, state_size])

        # Store experience in memory
        remember(state, action, reward, next_state, done)

        # Update state
        state = next_state

        if done:  # If episode ends
            print(f"Episode: {e+1}/{episodes}, Score: {time}, Epsilon: {epsilon:.2}")
            break

    # Train the model using replay memory
    if len(memory) > batch_size:
        replay(batch_size)

env.close()

# Step 7: Evaluate the performance

# Evaluation loop
evaluation_episodes = 10  # Number of evaluation episodes
scores = []  # Track scores for performance metrics

for e in range(evaluation_episodes):
    state = env.reset()
    if isinstance(state, tuple):  # Handle tuple output
        state = state[0]
    state = np.reshape(state, [1, state_size])

    total_reward = 0  # Track total reward per episode

    for time in range(200):  # Max steps per episode
        # Choose the greedy action
        action = np.argmax(model.predict(state)[0])

        # Perform action in the environment
        result = env.step(action)
        if len(result) == 4:  # Handle 4-value output
            next_state, reward, done, _ = result
        else:  # Handle 5-value output
            next_state, reward, done, _, _ = result

        if isinstance(next_state, tuple):  # Handle tuple next_state
            next_state = next_state[0]
        next_state = np.reshape(next_state, [1, state_size])

        state = next_state
        total_reward += reward

        if done:  # If episode ends
            print(f"Evaluation Episode: {e+1}/{evaluation_episodes}, Score: {time}, Total Reward: {total_reward}")
            scores.append(total_reward)
            break

# Summary of evaluation performance
print(f"Average Reward: {np.mean(scores):.2f}, Max Reward: {np.max(scores)}, Min Reward: {np.min(scores)}")

env.close()

# Exercise 1: Modify the Reward Function to Encourage Longer Episodes


import os

# Create sample directory structure if it does not exist
base_dir = 'sample_data'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')
class1_train = os.path.join(train_dir, 'class1')
class2_train = os.path.join(train_dir, 'class2')
class1_val = os.path.join(val_dir, 'class1')
class2_val = os.path.join(val_dir, 'class2')

# Create directories if they do not exist
for dir_path in [train_dir, val_dir, class1_train, class2_train, class1_val, class2_val]:
    os.makedirs(dir_path, exist_ok=True)

print("Directory structure created. Add your images to these directories.")

# Import the necessary library
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Modify data generator to include validation data
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'sample_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'sample_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Function to modify the reward to encourage longer episodes
def modify_reward(reward, next_state):
    # Penalize large pole angles
    pole_angle = abs(next_state[2])  # Extract the pole angle from the state
    penalty = 1 if pole_angle > 0.1 else 0  # Apply penalty if angle is large
    return reward - penalty  # Adjust reward

# Inside the training loop
# Example usage in a reinforcement learning training loop:
# reward = modify_reward(reward, next_state)  # Use the modified reward

# Exercise 2: Implement Early Stopping Based on Episode Length

# Import necessary libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Modify data generator to include validation data
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'sample_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'sample_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Early stopping parameters
consecutive_success_threshold = 100
success_episode_length = 195
consecutive_success_count = 0
episode_lengths = []  # Initialize episode lengths list

# Example of training loop (this should be your actual loop)
for episode in range(1000):  # Replace with actual loop condition
    # Training logic goes here
    episode_length = 200  # Example value, replace with actual calculation
    episode_lengths.append(episode_length)

    # Early stopping check
    if len(episode_lengths) > consecutive_success_threshold and all(
        length >= success_episode_length for length in episode_lengths[-consecutive_success_threshold:]
    ):
        print("Early stopping: Agent consistently reaches max episode length.")
        break  # This break is now correctly inside the loop

# Exercise 3: Experiment with Different Exploration Strategies

# Modify data generator to include validation data
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'sample_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'sample_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

def decay_epsilon(epsilon, episode, switch_episode=100):
    if episode < switch_episode:
        return max(epsilon - 0.01, 0.01)  # Linear decay
    else:
        return max(epsilon * 0.99, 0.01)  # Exponential decay

# Inside the training loop
epsilon = decay_epsilon(epsilon, e)  # Adjust epsilon based on the current episode


