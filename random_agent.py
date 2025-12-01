import random # For selecting random move (an action)
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