import torch
import random
import numpy as np
from env import Connect4Env
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt

# Function for evaluating performance of the RL agent
def evaluate_agent(num_games=100, print_boards=True):

    env = Connect4Env()
    agent = DQNAgent()   # Make sure this disables epsilon-greedy
    agent.model.load_state_dict(torch.load("connect4_dqn_model.pth"))
    agent.model.eval()
    agent.epsilon = 0.0 # disable epsilon-greedy

    wins = 0
    losses = 0
    draws = 0

    final_boards = []

    for g in range(num_games):
        state = env.reset()
        done = False

        while not done:

            # Agent move
            action = agent.act(state, env.available_actions())
            next_state, reward, done = env.step(action)
            state = next_state

            if done:
                if reward == 1:
                    wins += 1
                else:
                    draws += 1
                break

            # Opponent move (random)
            opp_action = random.choice(env.available_actions())
            next_state, opp_reward, done = env.step(opp_action)
            state = next_state

            if done:
                if opp_reward == 1:
                    losses += 1
                else:
                    draws += 1
                break

        # Store final board for debugging
        final_boards.append(env.board.copy())

    # Print game results
    print("\n=== Evaluation Results ===")
    print(f"Wins:   {wins}/{num_games}  ({wins/num_games:.2%})")
    print(f"Losses: {losses}/{num_games} ({losses/num_games:.2%})")
    print(f"Draws:  {draws}/{num_games}  ({draws/num_games:.2%})")

    if print_boards:
        print("\n=== Example Final Boards ===")
        for i in range(3):
            print(f"\nGame {i+1} Final Board:")
            print(final_boards[i])

    return wins, losses, draws, final_boards



# plot reward curve
#def plot_rewards(reward_list):
    #plt.figure(figsize=(8, 4))
    #plt.plot(reward_list)
    #plt.title("Training Reward Curve")
    #plt.xlabel("Episode")
    #plt.ylabel("Episode Reward")
    #plt.grid(True)
    #plt.show()



if __name__ == "__main__":
    # Run evaluation
    evaluate_agent(num_games=50, print_boards=True)

    # If you saved rewards: plot_rewards(all_rewards)
