import numpy as np
import tkinter as tk
from tkinter import messagebox

def board_to_tensor(board):
    # This function needs to be implemented based on your model requirements
    pass

def is_win(board, player):
    # Check rows, columns and diagonals
    for i in range(3):
        if np.all(board[i, :] == player) or np.all(board[:, i] == player):
            return True
    if np.all(np.diag(board) == player) or np.all(np.diag(np.fliplr(board)) == player):
        return True
    return False

def get_valid_moves(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i, j] == 0]

class TicTacToeGUI:
    def __init__(self, model):
        self.window = tk.Tk()
        self.window.title("Tic Tac Toe vs AI")
        self.model = model
        self.board = np.zeros((3, 3), dtype=int)
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        self.game_over = False
        
        for i in range(3):
            for j in range(3):
                self.buttons[i][j] = tk.Button(
                    self.window,
                    text="",
                    font=('Arial', 20),
                    width=5,
                    height=2,
                    command=lambda row=i, col=j: self.make_move(row, col)
                )
                self.buttons[i][j].grid(row=i, column=j)
        
        # Bot plays first
        self.make_bot_move()
    
    def make_move(self, i, j):
        if self.board[i][j] == 0 and not self.game_over:
            # Human move
            self.board[i][j] = -1
            self.buttons[i][j].config(text="O")
            
            if self.check_game_state(-1):
                return
            
            # Bot move
            self.make_bot_move()
    
    def make_bot_move(self):
        move = self.get_bot_move()
        if move:
            i, j = move
            self.board[i][j] = 1
            self.buttons[i][j].config(text="X")
            self.check_game_state(1)
    
    def get_bot_move(self):
        valid_moves = []
        safe_moves = []
        max_safe_score = -1
        best_safe_move = None
        max_any_score = -1
        best_any_move = None

        # Check all possible moves
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:
                    # Create hypothetical board
                    new_board = self.board.copy()
                    new_board[i, j] = 1

                    # Check if opponent can win next turn
                    opponent_can_win = False
                    for oi, oj in get_valid_moves(new_board):
                        opp_board = new_board.copy()
                        opp_board[oi, oj] = -1
                        if is_win(opp_board, -1):
                            opponent_can_win = True
                            break

                    # Evaluate model score
                    tensor = board_to_tensor(new_board)
                    score = self.model(tensor).item()

                    # Categorize moves
                    if not opponent_can_win:
                        safe_moves.append((i, j))
                        if score > max_safe_score:
                            max_safe_score = score
                            best_safe_move = (i, j)
                    if score > max_any_score:
                        max_any_score = score
                        best_any_move = (i, j)

        # Prioritize safe moves with highest score
        if best_safe_move:
            return best_safe_move

        # Fallback to best overall move (even if risky)
        return best_any_move
    
    def check_game_state(self, player):
        if is_win(self.board, player):
            winner = "Bot" if player == 1 else "You"
            messagebox.showinfo("Game Over", f"{winner} win!")
            self.game_over = True
            return True
        elif np.all(self.board != 0):
            messagebox.showinfo("Game Over", "It's a draw!")
            self.game_over = True
            return True
        return False
    
    def run(self):
        self.window.mainloop()

def play_tic_tac_toe(model):
    game = TicTacToeGUI(model)
    game.run()

# Load trained model
# model = ... # Your model loading code here
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WinNet(nn.Module):
    """Predicts W'(P, t) for a game state P."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(9, 64),  # Input: flattened 3x3 board
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output ∈ [0,1]
        )

    def forward(self, x):
        return self.fc(x)

def board_to_tensor(board):
    """Convert 3x3 board to tensor (flattened)."""
    return torch.tensor(board.flatten(), dtype=torch.float32, device=device)

# Start game
if __name__ == "__main__":
    model = WinNet().to(device)
    model.load_state_dict(torch.load("weights.pth", map_location=device))
    model.eval()
    play_tic_tac_toe(model)

# Run the script and play Tic Tac Toe against the AI model. The model will make its move based on the predicted win probability for each possible move. You can train the model using the provided training script and then use the trained model for inference in the game.