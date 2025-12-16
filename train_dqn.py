import torch
from env import Connect4Env
from dqn_agent import DQNAgent
import random
import matplotlib.pyplot as plt 

def train_dqn(
    episodes=2000, # how many games will be played
    batch_size=64, # how many experiences will be sampled each training step
    target_update_interval=50 # how often training network copies weights from main network
):

    env = Connect4Env()
    agent = DQNAgent()

    all_rewards = []
    all_losses = [] 

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

          loss = agent.train_step(batch_size)   # capture loss
          if loss is not None:                   
              all_losses.append(loss)         

          state = next_state
          total_reward += reward

          # If opponent wins or game ends, loop stops
          if done:
              break

          # opponent move random (random)
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

    # plot loss curve   
    plt.figure(figsize=(8, 4))  
    plt.plot(all_losses)        
    plt.xlabel("Training Step") 
    plt.ylabel("Loss")          
    plt.title("DQN Training Loss Over Time")  
    plt.grid(True)             
    plt.show()                 

    return all_rewards


if __name__ == "__main__":
    train_dqn()
