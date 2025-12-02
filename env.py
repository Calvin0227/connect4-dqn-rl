# data type is int
import numpy as np

# reset - reset board
# step - action,return next_state, reward, done
# available - return availabel action
# check_winder - check win / lose
# print_board - print board, for debug

class Connect4Env:
    def __init__(self, rows=6, cols=7):

        # Board is a matrix:
        #  1 = agent piece
        # -1 = opponent piece
        #  0 = empty
        self.rows = rows
        self.cols = cols
        self.board = None
        self.current_player = 1 # 1 = agent, -1 = opponent

    def reset(self):
        """ 
        Reset the board for a new episode and return state
        """
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1   # have the agent start first
        return self.board.copy()

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
            # so DQN will learn not to make random move
            return self.board.copy(), -2, False

        # 2. Drop the piece into the selected column
        # find the first available space starting from the bottom
        for r in range(self.rows - 1, -1, -1):
            if self.board[r][action] == 0:
                self.board[r][action] = self.current_player
                break

        # 3. Check if this move wins the game
        winner = self.check_winner()
        if winner == self.current_player:
            # Current player won
            return self.board.copy(), 1, True

        # 4. Check draw (board full)
        if len(self.available_actions()) == 0:
            return self.board.copy(), 0, True  # draw

        # 5. Switch player for next turn (make sure the game is turn-based)
        self.current_player *= -1

        # 6. Continue game with no reward
        return self.board.copy(), 0, False

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
