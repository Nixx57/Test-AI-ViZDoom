import os
from pathlib import Path
import vizdoom as vzd
import utils
import torch

# Paths
ROOT_DIR = Path(__file__).parent
MODEL_SAVE_PATH = ROOT_DIR / "models"
LOG_PATH = ROOT_DIR / "logs"
CURRICULUM_SAVE_PATH = ROOT_DIR / "curriculum"

# Create directories
for path in [MODEL_SAVE_PATH, LOG_PATH, CURRICULUM_SAVE_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# Game Configuration
VIZDOOM_MODE = vzd.Mode.PLAYER
SCREEN_RESOLUTION = vzd.ScreenResolution.RES_320X200
SCREEN_FORMAT = vzd.ScreenFormat.RGB24
EPISODE_COUNT = 10000  # Total number of training episodes
SOUND = False
WINDOW_VISIBLE = True

# Action Configuration
AVAILABLE_BUTTONS = [
    vzd.Button.ATTACK,
    vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT,
    vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT,
    vzd.Button.MOVE_FORWARD, vzd.Button.MOVE_BACKWARD,
    vzd.Button.USE,
    vzd.Button.SPEED,
    vzd.Button.SELECT_NEXT_WEAPON, vzd.Button.SELECT_PREV_WEAPON
]

# Special actions (one at a time)
SINGLE_ACTION_BUTTONS = [
    vzd.Button.USE,
]

# Combinable actions
COMBINABLE_BUTTONS = [
    vzd.Button.ATTACK,
    vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT,
    vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT,
    vzd.Button.MOVE_FORWARD, vzd.Button.MOVE_BACKWARD,
    vzd.Button.SPEED
]

# Exclusive action groups
EXCLUSIVE_BUTTON_GROUPS = [
    (vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT),
    (vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT),
    (vzd.Button.MOVE_FORWARD, vzd.Button.MOVE_BACKWARD),
    (vzd.Button.SELECT_NEXT_WEAPON, vzd.Button.SELECT_PREV_WEAPON),
]

# Game variables to track
GAME_VARIABLES = [
    vzd.GameVariable.KILLCOUNT,         # 0
    vzd.GameVariable.ITEMCOUNT,         # 1
    vzd.GameVariable.SECRETCOUNT,       # 2
    vzd.GameVariable.POSITION_X,        # 3
    vzd.GameVariable.POSITION_Y,        # 4
    vzd.GameVariable.POSITION_Z,        # 5
    vzd.GameVariable.ANGLE,             # 6
    vzd.GameVariable.HEALTH,            # 7
    vzd.GameVariable.ARMOR,             # 8
    vzd.GameVariable.ON_GROUND,         # 9
    vzd.GameVariable.ATTACK_READY,      # 10
    vzd.GameVariable.SELECTED_WEAPON,   # 11
    vzd.GameVariable.SELECTED_WEAPON_AMMO, # 12
    vzd.GameVariable.AMMO2,             # 13
    vzd.GameVariable.AMMO3,             # 14
    vzd.GameVariable.AMMO4,             # 15
    vzd.GameVariable.AMMO5,             # 16
    vzd.GameVariable.WEAPON0,           # 17
    vzd.GameVariable.WEAPON1,           # 18
    vzd.GameVariable.WEAPON2,           # 19
    vzd.GameVariable.WEAPON3,           # 20
    vzd.GameVariable.WEAPON4,           # 21
    vzd.GameVariable.WEAPON5,           # 22
    vzd.GameVariable.WEAPON6,           # 23
    vzd.GameVariable.WEAPON7,           # 24
    vzd.GameVariable.WEAPON8,           # 25
    vzd.GameVariable.WEAPON9,           # 26
]

# Generate action space
ACTIONS_LIST = utils.getAllActions()

# Reward Configuration
REWARD_LIVING = 0.1           # Bonus for staying alive
REWARD_KILL = 50.0            # Reward for killing an enemy
REWARD_ITEM_PICKUP = 30.0     # Reward for picking up an item
REWARD_WEAPON_PICKUP = 50.0   # Reward for picking up a weapon
REWARD_ARMOR_PICKUP = 20.0    # Reward for picking up armor
REWARD_HEALTH_PICKUP = 15.0   # Reward for picking up health
REWARD_AMMO_PICKUP = 2.0      # Reward for picking up ammo
REWARD_SECRET = 200.0         # Reward for finding a secret
REWARD_EXPLORATION = 20.0     # Reward for exploring
REWARD_DEATH = -100.0         # Penalty for dying
REWARD_OBJECTIVE_REACHED = 500.0  # Reward for reaching objective

# Training Hyperparameters
PPO_LEARNING_RATE = 0.00025
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_EPSILON = 0.2
PPO_EPOCHS = 10
PPO_BATCH_SIZE = 64

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def validate_config():
    """Validate the configuration"""
    # Ensure required directories exist
    for path in [MODEL_SAVE_PATH, LOG_PATH, CURRICULUM_SAVE_PATH]:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

# Auto-validate on import
validate_config()