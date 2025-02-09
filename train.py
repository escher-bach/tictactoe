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

def generate_data(num_samples=1000):
    """Generates labeled data for W(P, t) (ground truth)."""
    data = []
    for _ in range(num_samples):
        board = np.zeros((3, 3), dtype=int)
        # Simulate random game states
        num_moves = np.random.randint(0, 5)
        player = 1
        for _ in range(num_moves):
            valid_moves = [(i, j) for i in range(3) for j in range(3) if board[i, j] == 0]
            if not valid_moves:
                break
            move = valid_moves[np.random.choice(len(valid_moves))]
            board[move] = player
            player = -player

        # Determine ground truth W(P, t)
        if is_win(board, 1):
            label = 1.0  # Player 1 wins
        elif is_win(board, -1):
            label = 0.0  # Player 1 loses
        else:
            label = 0.5  # Ongoing game
        data.append((board.copy(), label))
    return data

def is_win(board, player):
    """Check if player has won."""
    for i in range(3):
        if all(board[i, :] == player) or all(board[:, i] == player):
            return True
    if all(np.diag(board) == player) or all(np.diag(np.fliplr(board)) == player):
        return True
    return False


# Get valid moves for a player
def get_valid_moves(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i, j] == 0]

# Apply a move to the board
def apply_move(board, move, player):
    new_board = board.copy()
    new_board[move] = player
    return new_board

def apply_random_move(board, player):
    """Applies a random valid move to the board for the given player."""
    valid_moves = get_valid_moves(board)
    if not valid_moves:
        return board
    move = valid_moves[np.random.choice(len(valid_moves))]
    return apply_move(board, move, player)

def constraint_loss(model, data):
    """Computes loss for all 4 conditions."""
    loss = 0.0

    # Split data into winning (W=1), losing (W=0), and normal (W=0.5)
    winning = [d for d in data if d[1] == 1.0]
    losing = [d for d in data if d[1] == 0.0]
    normal = [d for d in data if d[1] == 0.5]

    # Condition 1: W=1 → W'=1
    for board, _ in winning:
        w_pred = model(board_to_tensor(board))
        loss += (w_pred - 1.0) ** 2  # MSE

    # Condition 2: W=0 → W'(P) < W'(P')
    for board, _ in losing:
        w_pred_p = model(board_to_tensor(board))
        # Generate opponent's next move
        opp_board = apply_random_move(board, player=-1)
        w_pred_opp = model(board_to_tensor(opp_board))
        loss += torch.relu(w_pred_p - w_pred_opp + 0.1)  # Margin of 0.1

    # Condition 3: Valid transitions imply subsequent moves
    for board, _ in normal:
        valid_moves = get_valid_moves(board)
        if valid_moves:
            # Sample a valid next state P''
            move = valid_moves[np.random.choice(len(valid_moves))]
            next_board = apply_move(board, move, player=1)
            w_pred_p = model(board_to_tensor(board))
            w_pred_next = model(board_to_tensor(next_board))
            if w_pred_next >= w_pred_p:
                # Ensure subsequent valid moves exist
                next_valid = get_valid_moves(next_board)
                if next_valid:
                    # Check all P''' from P''
                    w_pred_next_next = [model(board_to_tensor(apply_move(next_board, m, -1))) for m in next_valid]
                    loss += torch.relu(1.0 - max(w_pred_next_next))  # Encourage at least one valid W'

    # Condition 4: Margin enforcement
    margin_loss = 0.0
    if len(winning) >= 2:
        # Sample two winning positions
        b1, b2 = winning[np.random.choice(len(winning), 2, replace=False)]
        w1 = model(board_to_tensor(b1[0]))
        w2 = model(board_to_tensor(b2[0]))
        # Difference between winning positions
        win_diff = torch.abs(w1 - w2)
        # Difference between winning and losing
        b_lose = losing[np.random.choice(len(losing))]
        w_lose = model(board_to_tensor(b_lose[0]))
        lose_diff = torch.abs(w1 - w_lose)
        margin_loss += torch.relu(lose_diff - win_diff + 0.1)  # Enforce win_diff > lose_diff
    loss += margin_loss

    return loss / len(data)

model = WinNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Generate training data
data = generate_data(num_samples=1000)

# Training
for epoch in range(100):
    optimizer.zero_grad()
    loss = constraint_loss(model, data)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")