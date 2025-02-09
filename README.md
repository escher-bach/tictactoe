# Constraint-Based Tic-Tac-Toe AI

This project implements a Tic-Tac-Toe AI using a novel constraint-based learning approach inspired by Łukasiewicz logic, rather than traditional reinforcement learning methods.

## Technical Approach

### Core Concept
Instead of using reinforcement learning or minimax algorithms, this AI learns through a constraint satisfaction approach where the winning probability function W'(P,t) is learned directly using a neural network. This approach is inspired by many-valued Łukasiewicz logic, where truth values can be any real number between 0 and 1.

### Win Probability Function
The neural network learns a function W'(P,t) that estimates the probability of winning from any given board position P. This function satisfies several key constraints:

1. W'(P) = 1 for winning positions
2. W'(P) < W'(P') for losing positions, where P' is the opponent's next move
3. Valid transitions must imply subsequent moves exist
4. Margin enforcement between winning and losing positions

### Training Constraints
The training uses a custom constraint loss function that enforces these logical properties:

- For winning positions (W=1), the network is trained to output exactly 1
- For losing positions (W=0), the network ensures the current position has a lower score than opponent's next move
- For ongoing games, the network maintains valid transition properties
- A margin is enforced between winning and losing positions to create clear decision boundaries

### Network Architecture
- Simple feedforward neural network with 9 inputs (flattened board state)
- Hidden layer with 64 neurons and ReLU activation
- Output layer with sigmoid activation to bound predictions between [0,1]

## Usage

1. Train the model:
```bash
python train.py
```

2. Play against the trained AI:
```bash
python infer.py
```

## Implementation Details

The constraint loss function implements four key conditions:
1. Winning positions must have W'(P) = 1
2. Losing positions must have lower scores than opponent's next moves
3. Valid transitions must maintain logical consistency
4. Margin enforcement between different game states

This approach differs from traditional reinforcement learning by:
- Not requiring episode-based training
- Learning directly from game state constraints
- Using supervised learning with custom loss function
- Incorporating logical rules inspired by Łukasiewicz logic

## Requirements
- Python 3.x
- PyTorch
- NumPy
- Tkinter (for GUI)