import numpy as np
import torch
import torch.nn as nn # NN layers
import torch.nn.functional as F # activation and loss functions for NN

class Connect4Env:
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        
        # Board is a matrix:
        #  1 = agent piece
        # -1 = opponent piece
        #  0 = empty
        self.board = None
        
        # Player whose turn it is (agent always starts by default)
        self.current_player = 1

    # Preprocessor
    def board_to_cnn_input(self, board):
      # make sure the board given is converted into a numpy array
      board = np.array(board, dtype=np.float32)

      # Make channel for player's pieces (1)
      player_channel = (board == 1).astype('float32')

      # Make channel for opponent's pieces (-1)
      opponent_channel = (board == -1).astype('float32')

      stacked = np.stack([player_channel, opponent_channel], axis=0)  # (2, 6, 7)
      return stacked

    def reset(self):
        """ 
        Reset the board for a new episode. 
        """
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1   # have the agent start first
        return self.board_to_cnn_input(self.board)

    def available_actions(self):
        """
        Return list of columns where a move is possible.
        Only columns whose top cell is empty are legal.
        """
        return [c for c in range(self.cols) if self.board[0][c] == 0]

    # The function will perform actions in the Connect 4 game
    def step(self, action):
        """
        Take one action (drop piece in column).
        Returns: new_state, reward, and done attributes.
        """

        # 1. Handle illegal move
        if action not in self.available_actions():
            # Penalize but not end the game, which will help with RL
            return self.board_to_cnn_input(self.board), -2, False

        # 2. Drop the piece into the selected column
        for r in range(self.rows - 1, -1, -1):
            if self.board[r][action] == 0:
                self.board[r][action] = self.current_player
                break

        # 3. Check if this move wins the game
        winner = self.check_winner()
        if winner == self.current_player:
            # Current player won
            return self.board_to_cnn_input(self.board), 1, True

        # 4. Check draw (board full)
        if len(self.available_actions()) == 0:
            return self.board_to_cnn_input(self.board), 0, True  # draw

        # 5. Switch player for next turn (make sure the game is turn-based)
        self.current_player *= -1

        # 6. Continue game with no reward
        return self.board_to_cnn_input(self.board), 0, False

    def check_winner(self):
        """
        Check board for a 4-in-a-row.
        Returns 1 (agent), -1 (opponent), or 0 (nobody).
        """
        b = self.board

        # Horizontal check
        for r in range(self.rows):
            for c in range(self.cols - 3):
                window = b[r, c:c+4]
                if abs(sum(window)) == 4:  # means 4 same pieces
                    return window[0]

        # Vertical check
        for c in range(self.cols):
            for r in range(self.rows - 3):
                window = b[r:r+4, c]
                if abs(sum(window)) == 4:
                    return window[0]

        # Diagonal (down-right)
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                window = [b[r+i][c+i] for i in range(4)]
                if abs(sum(window)) == 4:
                    return window[0]

        # Diagonal (up-right)
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                window = [b[r-i][c+i] for i in range(4)]
                if abs(sum(window)) == 4:
                    return window[0]

        return 0  # no winner 

    def print_board(self):
        """display board after every turn in."""
        print("\nBoard:")
        for r in range(self.rows):
            line = ""
            for c in range(self.cols):
                val = self.board[r][c]
                if val == 1:
                    line += " X "
                elif val == -1:
                    line += " O "
                else:
                    line += " . "
            print(line)
        print()

        ###################################################################################
        import random
#import numpy as np
from collections import deque

# Data structure that stores all experiences (we will use these experiences, which are just tuples of features, as training data for DQN)
class ReplayBuffer:
    # construct a replay buffer (double sided queue = deque) with a maximum capacity of 50,000 experiences
    def __init__(self, capacity=50000): 
        self.buffer = deque(maxlen=capacity)

    # store experience tuples in the replay buffer
    def push(self, state, action, reward, next_state, done):
        # Convert to numpy arrays
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        self.buffer.append((state, action, reward, next_state, done))

    # randomly sample batch_size amount of experiences from the buffer and store in batch
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # Convert all attributes to numpy arrays
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)
    ###################################################################################
    import random # selecting random move (an action)
from env import Connect4Env # importing the connect 4 environment so it can be used
from replay_buffer import ReplayBuffer # importing the replay buffer to store experiences

def random_agent(env, state):
    # return random legal move out of all available legal actions
    return random.choice(env.available_actions())


# Run a full game
if __name__ == "__main__":

    
    env = Connect4Env() # create a connect 4 environment 
    state = env.reset() # make sure we have a empty board
    done = False # we know game just started so its not done
    step_count = 0 # represents number of moves that has occured

    # Construct a replay buffer so that the experiences can be stored 
    replay_buffer = ReplayBuffer(capacity=50000)

    while not done:

        # have agent select a random legal move
        action = random_agent(env, state)

        # agent performs that random legal move
        next_state, reward, done = env.step(action)

        # store the attribute values in the replay buffer
        replay_buffer.push(state, action, reward, next_state, done)

        # description of step number, current player, action performed
        print(f"Step {step_count}, Player: {env.current_player * -1}, Action: {action}")
         
        env.print_board() # display board
        state = next_state # update the state
        step_count += 1 

    # Game is done so display the results
    print("Game over!")
    print("Replay buffer now contains:", len(replay_buffer), "experiences")
    winner = env.check_winner()
    if winner == 1:
        print("Winner: X (Player 1)")
    elif winner == -1:
        print("Winner: O (Player -1)")
    else:
        print("Draw!")
        ###################################################################################

    import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from replay_buffer import ReplayBuffer

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

        # construct a connect4DQN model for main network
        self.model = Connect4DQN(num_actions)
        # construct a connect4DQN model for target network
        self.target_model = Connect4DQN(num_actions)

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

        # convert to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # (1, 2, 6, 7)

        q_values = self.model(state_tensor).detach().numpy()[0]

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

        # convert to tensor
        states = torch.FloatTensor(states)          
        next_states = torch.FloatTensor(next_states)


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

    
    # Update target network occasionally
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
        ###################################################################################
        import torch
#from env import Connect4Env
#from dqn_agent import DQNAgent

def train_dqn(
    episodes=2000, # how many games will be played
    batch_size=64, # how many experiences will be sampled each training step
    target_update_interval=50 # how often training network copies weights from main network
):

    env = Connect4Env()
    agent = DQNAgent()

    all_rewards = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
      
          
          # 1. AGENT MOVE
          action = agent.act(state, env.available_actions())
          next_state, reward, done = env.step(action)

          # store agent's experience
          agent.remember(state, action, reward, next_state, done)
          agent.train_step(batch_size)

          state = next_state
          total_reward += reward

          # If opponent wins or game ends, loop stops
          if done:
              break

          # 2. OPPONENT MOVE (RANDOM)
          opp_action = random.choice(env.available_actions())
          next_state, opp_reward, done = env.step(opp_action)

          # know that opponent reward is bad for agent
          agent.remember(state, opp_action, -opp_reward, next_state, done)

          # know that opponent does not train the network.
          state = next_state

        # Update target network
        if ep % target_update_interval == 0:
            agent.update_target()

        all_rewards.append(total_reward)

        print(f"Episode {ep}/{episodes} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

    # save model
    torch.save(agent.model.state_dict(), "connect4_dqn_model.pth")
    print("\nTraining finished. Model saved as connect4_dqn_model.pth")

    return all_rewards


if __name__ == "__main__":
    train_dqn()