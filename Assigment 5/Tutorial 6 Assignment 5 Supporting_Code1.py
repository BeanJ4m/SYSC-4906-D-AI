import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib
matplotlib.use('QtAgg')  # Interactive backend
import matplotlib.pyplot as plt


# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Environment Setup
n_channels = 5
n_episodes = 10000
true_probs = [0.1, 0.4, 0.6, 0.3, 0.9]

# Q-Learning Parameters
alpha = 0.1
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.999

# Initialize Q-table for Q-learning
Q_table = np.zeros(n_channels)

# DQN Hyperparameters
alpha_dqn = 0.0005
batch_size = 128
memory_size = 50000
target_update_freq = 20

# Define the Deep Q-Network (DQN)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

# Initialize Neural Networks and optimizer for DQN
policy_net = DQN(n_channels, n_channels)
target_net = DQN(n_channels, n_channels)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=alpha_dqn)
criterion = nn.MSELoss()

# Experience Replay Memory for DQN
memory = deque(maxlen=memory_size)

# Lists to store rewards
rewards_qlearning = []
rewards_dqn = []

# Training Loop
for episode in range(n_episodes):

    # Q-Learning (epsilon-greedy action selection)
    if np.random.rand() < epsilon:
        action_qlearning = np.random.choice(n_channels)
    else:
        action_qlearning = np.argmax(Q_table)

    reward_qlearning = 1 if np.random.rand() < true_probs[action_qlearning] else 0

    # Q-table update (Q-learning rule)
    Q_table[action_qlearning] += alpha * (reward_qlearning - Q_table[action_qlearning])

    rewards_qlearning.append(reward_qlearning)

    # DQN (epsilon-greedy action selection)
    state_dqn = torch.eye(n_channels)[0]  # static state representation (single state scenario)
    if np.random.rand() < epsilon:
        action_dqn = np.random.choice(n_channels)
    else:
        with torch.no_grad():
            action_dqn = policy_net(state_dqn).argmax().item()

    reward_dqn = 1 if np.random.rand() < true_probs[action_dqn] else 0

    # Store experiences in replay memory (state, action, reward, next_state)
    memory.append((state_dqn.numpy(), action_dqn, reward_dqn, state_dqn.numpy()))

    rewards_dqn.append(reward_dqn)

    # DQN training using replay memory
    if len(memory) >= batch_size:
        batch_samples = random.sample(memory, batch_size)
        states_batch, actions_batch, rewards_batch, next_states_batch = zip(*batch_samples)

        states_batch_tensor = torch.from_numpy(np.array(states_batch)).float()
        actions_batch_tensor = torch.LongTensor(actions_batch).unsqueeze(1)
        rewards_batch_tensor = torch.FloatTensor(rewards_batch).unsqueeze(1)
        next_states_batch_tensor = torch.from_numpy(np.array(next_states_batch)).float()

        # Compute current Q-values and target Q-values
        current_Q_values = policy_net(states_batch_tensor).gather(1, actions_batch_tensor)
        next_Q_values_target_net = target_net(next_states_batch_tensor).detach().max(1)[0].unsqueeze(1)
        target_Q_values = rewards_batch_tensor + gamma * next_Q_values_target_net

        # Compute loss and update DQN parameters
        loss = criterion(current_Q_values, target_Q_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Update target network periodically
    if episode % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Decay epsilon after each episode until minimum is reached
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

# Moving Average Calculation Function:
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Plot Learning Curve for Q-Learning vs. DQN:
window_size_plotting = 200

plt.figure(figsize=(12,6))
plt.plot(moving_average(rewards_qlearning, window_size_plotting), label="Q-Learning")
plt.plot(moving_average(rewards_dqn, window_size_plotting), label="DQN")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Learning Curve Comparison: Q-Learning vs. DQN")
plt.legend()
plt.grid(True)
plt.show()
