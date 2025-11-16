import numpy as np

class Connect4Env:
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.board = None
        self.current_player = 1  # 1 = agent, -1 = opponent
    
    def reset(self):
        """Reset the board and return the initial state"""
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        return self.board.copy()

    def available_actions(self):
        """Return a list of columns that are not full"""
        actions = []
        for c in range(self.cols):
            if self.board[0][c] == 0:  # top row empty => column is playable
                actions.append(c)
        return actions

    def step(self, action):
        """Drop a piece into the selected column and update the board"""
        # 1. Validate action
        if action not in self.available_actions():
            # Illegal move: return big negative reward
            return self.board.copy(), -10, True

        # 2. Drop piece to the lowest available row
        for r in range(self.rows - 1, -1, -1):
            if self.board[r][action] == 0:
                self.board[r][action] = self.current_player
                break

        # 3. Check win
        winner = self.check_winner()
        if winner == self.current_player:
            return self.board.copy(), 1, True  # win reward

        # 4. Check draw
        if len(self.available_actions()) == 0:
            return self.board.copy(), 0, True  # draw

        # 5. Switch player
        self.current_player *= -1

        # 6. Continue game
        return self.board.copy(), 0, False

    def check_winner(self):
        """Return 1 or -1 if a player wins, else 0"""
        board = self.board

        # Horizontal check
        for r in range(self.rows):
            for c in range(self.cols - 3):
                window = board[r, c:c+4]
                if abs(sum(window)) == 4 and 0 not in window:
                    return window[0]

        # Vertical check
        for c in range(self.cols):
            for r in range(self.rows - 3):
                window = board[r:r+4, c]
                if abs(sum(window)) == 4 and 0 not in window:
                    return window[0]

        # Diagonal (down-right)
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                window = [board[r+i][c+i] for i in range(4)]
                if abs(sum(window)) == 4 and 0 not in window:
                    return window[0]

        # Diagonal (up-right)
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                window = [board[r-i][c+i] for i in range(4)]
                if abs(sum(window)) == 4 and 0 not in window:
                    return window[0]

        return 0

    def print_board(self):
        """Print board in human-friendly format"""
        print("\nBoard:")
        for r in range(self.rows):
            line = ""
            for c in range(self.cols):
                v = self.board[r][c]
                if v == 1:
                    line += " X "
                elif v == -1:
                    line += " O "
                else:
                    line += " . "
            print(line)
        print()
