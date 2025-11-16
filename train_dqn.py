import torch
from env import Connect4Env
from dqn_agent import DQNAgent

def train_dqn(
    episodes=2000,
    batch_size=64,
    target_update_interval=50
):

    env = Connect4Env()
    agent = DQNAgent()

    all_rewards = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            available_actions = env.available_actions()

            action = agent.act(state, available_actions)
            next_state, reward, done = env.step(action)

            agent.remember(state, action, reward, next_state, done)

            agent.train_step(batch_size)

            state = next_state
            total_reward += reward

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
