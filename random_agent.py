import random
from env import Connect4Env

def random_agent(env, state):
    return random.choice(env.available_actions())

if __name__ == "__main__":
    env = Connect4Env()
    state = env.reset()

    done = False
    step_count = 0

    while not done:
        action = random_agent(env, state)
        next_state, reward, done = env.step(action)

        print(f"Step {step_count}, Player: {env.current_player * -1}, Action: {action}")
        env.print_board()

        state = next_state
        step_count += 1

    print("Game over!")
    winner = env.check_winner()
    if winner == 1:
        print("Winner: X (Player 1)")
    elif winner == -1:
        print("Winner: O (Player -1)")
    else:
        print("Draw!")
