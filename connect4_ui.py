import tkinter as tk
from tkinter import messagebox
import torch
import random
from env import Connect4Env
from dqn_agent import DQNAgent

CELL_SIZE = 80
ROWS, COLS = 6, 7

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Connect4UI:
    def __init__(self):
        self.env = Connect4Env()
        
        # Game settings
        self.player_first = True  # Player goes first
        self.ai_difficulty = "random"  # "random" or "dqn"
        self.game_active = True
        
        # Load DQN model
        self.dqn_agent = DQNAgent()
        try:
            self.dqn_agent.model.load_state_dict(torch.load("connect4_dqn_model.pth", map_location=device))
            self.dqn_agent.model.eval()
            self.dqn_agent.epsilon = 0.0  # Disable exploration
            self.dqn_loaded = True
            print("DQN model loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load DQN model: {e}")
            self.dqn_loaded = False

        # Create main window
        self.window = tk.Tk()
        self.window.title("Connect 4 - DQN AI")
        self.window.configure(bg="#2c3e50")
        
        # ========== Top Control Panel ==========
        self.control_frame = tk.Frame(self.window, bg="#2c3e50", pady=10)
        self.control_frame.pack()
        
        # First move selection
        tk.Label(self.control_frame, text="First:", bg="#2c3e50", fg="white", 
                 font=("Arial", 12)).grid(row=0, column=0, padx=5)
        
        self.first_var = tk.StringVar(value="player")
        self.player_first_btn = tk.Radiobutton(
            self.control_frame, text="Player (Red)", variable=self.first_var, 
            value="player", bg="#2c3e50", fg="white", selectcolor="#34495e",
            font=("Arial", 11), command=self.on_first_change
        )
        self.player_first_btn.grid(row=0, column=1, padx=5)
        
        self.ai_first_btn = tk.Radiobutton(
            self.control_frame, text="AI (Yellow)", variable=self.first_var, 
            value="ai", bg="#2c3e50", fg="white", selectcolor="#34495e",
            font=("Arial", 11), command=self.on_first_change
        )
        self.ai_first_btn.grid(row=0, column=2, padx=5)
        
        # Separator
        tk.Label(self.control_frame, text="   |   ", bg="#2c3e50", fg="gray").grid(row=0, column=3)
        
        # Difficulty selection
        tk.Label(self.control_frame, text="AI Difficulty:", bg="#2c3e50", fg="white",
                 font=("Arial", 12)).grid(row=0, column=4, padx=5)
        
        self.difficulty_var = tk.StringVar(value="random")
        self.random_btn = tk.Radiobutton(
            self.control_frame, text="Random", variable=self.difficulty_var,
            value="random", bg="#2c3e50", fg="white", selectcolor="#34495e",
            font=("Arial", 11), command=self.on_difficulty_change
        )
        self.random_btn.grid(row=0, column=5, padx=5)
        
        self.dqn_btn = tk.Radiobutton(
            self.control_frame, text="DQN", variable=self.difficulty_var,
            value="dqn", bg="#2c3e50", fg="white", selectcolor="#34495e",
            font=("Arial", 11), command=self.on_difficulty_change
        )
        self.dqn_btn.grid(row=0, column=6, padx=5)
        
        if not self.dqn_loaded:
            self.dqn_btn.config(state="disabled")
        
        # ========== Status Display ==========
        self.status_frame = tk.Frame(self.window, bg="#2c3e50", pady=5)
        self.status_frame.pack()
        
        self.status_label = tk.Label(
            self.status_frame, text="", 
            bg="#2c3e50", fg="#ecf0f1", font=("Arial", 14, "bold")
        )
        self.status_label.pack()
        
        # Turn indicator
        self.turn_indicator = tk.Label(
            self.status_frame, text="", bg="#2c3e50", fg="#f39c12", 
            font=("Arial", 12)
        )
        self.turn_indicator.pack()

        # ========== Game Board ==========
        self.canvas = tk.Canvas(
            self.window,
            width=COLS * CELL_SIZE,
            height=ROWS * CELL_SIZE,
            bg="#3498db",
            highlightthickness=0
        )
        self.canvas.pack(pady=10)
        
        # Bind click event
        self.canvas.bind("<Button-1>", self.handle_click)
        
        # ========== Bottom Button ==========
        self.button_frame = tk.Frame(self.window, bg="#2c3e50", pady=10)
        self.button_frame.pack()
        
        self.restart_btn = tk.Button(
            self.button_frame, text="Restart", command=self.restart_game,
            font=("Arial", 12, "bold"), bg="#e74c3c", fg="white",
            width=12, height=2, cursor="hand2"
        )
        self.restart_btn.pack()
        
        # Start first game
        self.start_game()
        
        self.window.mainloop()

    def on_first_change(self):
        """When first move setting changes, restart game"""
        self.restart_game()

    def on_difficulty_change(self):
        """When difficulty setting changes, restart game"""
        self.restart_game()

    def start_game(self):
        """Start a new game"""
        # Read settings
        self.player_first = (self.first_var.get() == "player")
        self.ai_difficulty = self.difficulty_var.get()
        
        # Reset environment
        self.state = self.env.reset()
        self.game_active = True
        
        # Update status
        difficulty_text = "DQN" if self.ai_difficulty == "dqn" else "Random"
        self.status_label.config(
            text=f"AI Difficulty: {difficulty_text}", fg="#2ecc71"
        )
        
        self.draw_board()
        
        # If AI goes first, make AI move
        if not self.player_first:
            self.update_turn_indicator(is_player_turn=False)
            self.window.after(500, self.ai_move)
        else:
            self.update_turn_indicator(is_player_turn=True)

    def restart_game(self):
        """Restart the game"""
        self.start_game()

    def update_turn_indicator(self, is_player_turn):
        """Update turn indicator"""
        if is_player_turn:
            self.turn_indicator.config(text="🔴 Your Turn!", fg="#e74c3c")
        else:
            self.turn_indicator.config(text="🟡 AI Thinking...", fg="#f1c40f")

    def draw_board(self):
        """Draw the board"""
        self.canvas.delete("all")

        for r in range(ROWS):
            for c in range(COLS):
                x1 = c * CELL_SIZE + 10
                y1 = r * CELL_SIZE + 10
                x2 = x1 + CELL_SIZE - 20
                y2 = y1 + CELL_SIZE - 20

                val = self.env.board[r][c]
                if val == 1:
                    color = "#e74c3c"  # Red - Player
                    outline = "#c0392b"
                elif val == -1:
                    color = "#f1c40f"  # Yellow - AI
                    outline = "#d68910"
                else:
                    color = "white"
                    outline = "#bdc3c7"

                self.canvas.create_oval(
                    x1, y1, x2, y2, 
                    fill=color, outline=outline, width=2
                )

    def handle_click(self, event):
        """Handle player click"""
        if not self.game_active:
            return
            
        col = event.x // CELL_SIZE
        
        # Check if valid move
        if col not in self.env.available_actions():
            self.status_label.config(text="Column full! Choose another.", fg="#e74c3c")
            return

        # Player move
        next_state, reward, done = self.env.step(col)
        self.state = next_state
        self.draw_board()

        if done:
            self.game_over(reward, "player")
            return

        # AI turn
        self.update_turn_indicator(is_player_turn=False)
        self.window.after(400, self.ai_move)

    def ai_move(self):
        """AI makes a move"""
        if not self.game_active:
            return
            
        available = self.env.available_actions()
        if not available:
            return
        
        # Choose action based on difficulty
        if self.ai_difficulty == "dqn" and self.dqn_loaded:
            action = self.dqn_agent.act(self.state, available)
        else:
            action = random.choice(available)
        
        next_state, reward, done = self.env.step(action)
        self.state = next_state
        self.draw_board()

        if done:
            self.game_over(reward, "ai")
        else:
            self.update_turn_indicator(is_player_turn=True)
            difficulty_text = "DQN" if self.ai_difficulty == "dqn" else "Random"
            self.status_label.config(
                text=f"AI ({difficulty_text}) played column {action + 1}", fg="#3498db"
            )

    def game_over(self, reward, last_player):
        """Handle game over"""
        self.game_active = False
        self.turn_indicator.config(text="")
        
        if reward == 1:
            if last_player == "player":
                msg = "You Win!"
                color = "#2ecc71"
            else:
                msg = "AI Wins!"
                color = "#e74c3c"
        else:
            msg = "Draw!"
            color = "#f39c12"

        self.status_label.config(text=msg, fg=color)
        
        # Show message and auto restart after delay
        messagebox.showinfo("Game Over", msg)
        self.window.after(500, self.restart_game)


if __name__ == "__main__":
    Connect4UI()