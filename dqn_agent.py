import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from replay_buffer import ReplayBuffer

# Detect GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

### We are combining a Convolutional Neural Network with Q-learning to make a Deep Q-learning Network
class Connect4DQN(nn.Module):

    def __init__(self, num_actions=7): # we want to generate 7 Q-alues (represents an action taken in one of the 7 columns of connect4)

        # initialize the model
        super().__init__()

        # The input shape required is a tensor of (batch_size, 2, 6, 7)
        # 6 = height of board; 7 = width of board; 2 = # of channels (player and opponent); batch_size = # of examples used for every update

        # Convolutional component (cnn portion of the whole network)
        self.conv_layers = nn.Sequential(

            # Learn basic patterns (edges, adjacency, simple lines)
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=1),
            # provide non-linearity so that the model can learn more complicated patterns
            nn.ReLU(),

            # Learn more complex patterns (2-in-a-row, simple threats, etc.)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            # A refining layer for playing around more with all features already extracted
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Compute linear 1D input size to be given to the fully connected layer:
        self.flatten_size = 64 * 6 * 7

        # Fully Connected component (fully connected portion of the whole network)
        self.fc_layers = nn.Sequential(

            # transform input into a 1D structure so it can be used as input for fully connected portion
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)   # 7 output neurons for output layer (since we need 7 Q-values)
        )

    def forward(self, x):
        # x is shape (batch, 2, 6, 7)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)


# DQN Agent
class DQNAgent:
    def __init__(
        self,
        state_shape=(6, 7), # shape of the board
        num_actions=7, # number of possible actions
        lr=1e-3, # learning rate
        gamma=0.99, # dicount factor
        epsilon_start=1.0, # pure random
        epsilon_end=0.1, # minimum exploration
        epsilon_decay=0.995 # gradually reduce randomness
    ):
        self.num_actions = num_actions
        self.device = device  # Store device reference

        # construct a connect4DQN model for main network - MOVE TO GPU
        self.model = Connect4DQN(num_actions).to(device)
        # construct a connect4DQN model for target network - MOVE TO GPU
        self.target_model = Connect4DQN(num_actions).to(device)

        self.target_model.load_state_dict(self.model.state_dict())  # sync target

        # Use the Adam optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # Use MSE loss function
        self.loss_fn = nn.MSELoss()

        self.gamma = gamma

        # epsilon-greedy exploration (determines how random the moves are during training)
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # construct replay buffer and use as memory
        self.memory = ReplayBuffer(capacity=50000)


    # Choose action
    def act(self, state, available_actions):
        if random.random() < self.epsilon:
            return random.choice(available_actions)

        # convert to tensor - MOVE TO GPU
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, 2, 6, 7)

        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy()[0]  # Move back to CPU for numpy

        masked_q = {a: q_values[a] for a in available_actions}
        best_action = max(masked_q, key=masked_q.get)

        return best_action

    # Store a transition
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)


    # Train one step
    def train_step(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        # convert to tensor - MOVE TO GPU
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.model(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
        target = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay


    # Update target network occasionally
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())