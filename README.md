# Experimental ViZDoom AI Agent with Curriculum Learning

## ⚠️ Experimental Project Notice
This is an amateur implementation of a reinforcement learning agent for ViZDoom. This project is experimental in nature and comes with no guarantees of performance or functionality. Some features may be incomplete, unused variables might exist, and the agent's learning success is not guaranteed.

## Overview
This project is an experimental attempt to implement an AI agent that learns to play Doom using Deep Reinforcement Learning (PPO algorithm) with a custom reward system and curriculum learning. The agent is designed to learn through trial and error, gradually progressing from basic movement to more complex behaviors, though actual learning outcomes may vary.

## Implementation Attempts

This project includes experimental implementations of:

- **PPO Implementation**: An attempt at implementing Proximal Policy Optimization
- **Reward System**: Experimental reward calculations for various game behaviors
- **Curriculum Learning**: Basic implementation of progressive difficulty scaling
- **Tactical Behaviors**: Experimental implementations of combat strategies and exploration

⚠️ Note: These features are implemented as part of a learning exercise and may not function optimally.

## Project Structure

- `training.py`: Main training script and PPO agent implementation
- `reward_system.py`: Advanced reward calculation and game state tracking
- `curriculum_learning.py`: Manages the progression through difficulty levels
- `config.py`: Configuration parameters for training, rewards, and environment
- `utils.py`: Utility functions and action space generation

## Requirements

- Python 3.8+
- PyTorch
- ViZDoom
- NumPy
- OpenCV

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install vizdoom --pre numpy opencv-python torch
   ```
3. (optional)Set up Doom2.wad:
   - Place `doom2.wad` in the ViZDoom installation directory (typically in `site-packages/vizdoom`)
   - Remove `freedoom2.wad` if present to ensure the correct WAD file is used

## Usage

1. Configure your training parameters in `config.py`
2. Start training:
   ```
   python training.py
   ```
3. Monitor progress in the console output

## Training Process

The agent will progress through different levels of difficulty, starting with basic movement and gradually introducing more complex objectives. Training statistics and model checkpoints are saved automatically.

## Customization

- Adjust reward weights in `config.py`
- Modify the curriculum in `curriculum_learning.py`
- Tune hyperparameters in the training script

## Contributing

Contributions are welcome

## License

WTFPL (Do What The Fuck You Want To Public License)
