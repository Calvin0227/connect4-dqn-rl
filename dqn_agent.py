import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from replay_buffer import ReplayBuffer


# -------------------------
# Q-Network (a simple MLP)
# -------------------------
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# DQN Agent
# -------------------------
class DQNAgent:
    def __init__(
        self,
        state_shape=(6, 7),
        action_size=7,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995
    ):
        self.state_dim = state_shape[0] * state_shape[1]  # flatten 6x7 → 42 input
        self.action_size = action_size

        self.model = QNetwork(self.state_dim, action_size)
        self.target_model = QNetwork(self.state_dim, action_size)
        self.target_model.load_state_dict(self.model.state_dict())  # sync target

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.gamma = gamma

        # epsilon-greedy exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # replay buffer
        self.memory = ReplayBuffer(capacity=50000)

    # -----------
    # Choose action
    # -----------
    def act(self, state, available_actions):
        if random.random() < self.epsilon:
            return random.choice(available_actions)

        state = np.array(state).reshape(1, -1)
        state_tensor = torch.FloatTensor(state)

        q_values = self.model(state_tensor).detach().numpy()[0]

        masked_q = {a: q_values[a] for a in available_actions}
        best_action = max(masked_q, key=masked_q.get)

        return best_action

    # -----------
    # Store a transition
    # -----------
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    # -----------
    # Train one step
    # -----------
    def train_step(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        states = torch.FloatTensor(states.reshape(batch_size, -1))
        next_states = torch.FloatTensor(next_states.reshape(batch_size, -1))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        next_q = self.target_model(next_states).max(1)[0].detach()
        target = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    # -----------
    # Update target network occasionally
    # -----------
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
