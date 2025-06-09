#!/usr/bin/env python3

import inspect
import itertools as it
import math
import os
import random
from collections import deque, namedtuple
from time import sleep, time

import numpy as np
import skimage.transform
import torch
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import trange
import torch.nn.functional as F

import vizdoom as vzd


# Q-learning settings
learning_rate = 0.00025
discount_factor = 0.99
train_epochs = 5
learning_steps_per_epoch = 2000
replay_memory_size = 100000

# NN learning settings
batch_size = 64

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 12
resolution = (30, 45)
episodes_to_watch = 10

model_savefile = "./model-doom.pth"
save_model = True
load_model = True
skip_learning = False

Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done'])

# Configuration file path
config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios", "map02.cfg")

# Uses GPU if available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")


def preprocess(img):
    """Down samples image to resolution and normalizes"""
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
    return img


def create_simple_game():
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_doom_skill(4)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(config.SCREEN_RESOLUTION)
    game.set_available_buttons(config.AVAILABLE_BUTTONS)
    game.set_available_game_variables(config.GAME_VARIABLES)
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.set_automap_buffer_enabled(True)
    game.set_objects_info_enabled(True)
    game.set_sectors_info_enabled(True)
    game.set_render_hud(True)
    game.set_render_minimal_hud(False)
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(True)
    game.set_render_particles(True)
    game.set_render_effects_sprites(True)
    game.set_render_messages(True)
    game.set_render_corpses(True)
    game.set_render_screen_flashes(True)

    game.set_living_reward(config.REWARD_LIVING)
    game.set_death_reward(config.REWARD_DEATH)
    game.set_map_exit_reward(config.REWARD_MAP_EXIT)
    game.set_kill_reward(config.REWARD_KILL)
    game.set_frag_reward(config.REWARD_KILL)
    game.set_secret_reward(config.REWARD_SECRET)
    game.set_item_reward(config.REWARD_ITEM_PICKUP)
    game.set_damage_made_reward(config.REWARD_DAMAGE_MADE)
    game.set_hit_reward(config.REWARD_HIT)
    game.set_hit_taken_reward(config.REWARD_HIT_TAKEN)
    game.set_damage_taken_reward(config.REWARD_DAMAGE_TAKEN)
    game.init()
    print("Doom initialized.")

    return game


def test(game, agent):
    """Runs a test_episodes_per_epoch episodes and prints the result"""
    print("\nTesting...")
    test_scores = []
    for test_episode in trange(test_episodes_per_epoch, leave=False):
        game.new_episode()
        agent.reset_hidden()
        
        # Initialize frame stack for test (same as in training)
        frame_stack = deque(maxlen=4)
        initial_state = preprocess(game.get_state().screen_buffer)
        for _ in range(4):
            frame_stack.append(initial_state)
        
        while not game.is_episode_finished():
            # Use frame stack like in training
            state = np.stack(frame_stack, axis=0)
            best_action_index = agent.get_action(state)
            game.make_action(actions[best_action_index], frame_repeat)
            
            # Update frame stack if episode continues
            if not game.is_episode_finished():
                new_frame = preprocess(game.get_state().screen_buffer)
                frame_stack.append(new_frame)
                
        r = game.get_total_reward()
        test_scores.append(r)

    test_scores = np.array(test_scores)
    print(
        "Results: mean: {:.1f} +/- {:.1f},".format(
            test_scores.mean(), test_scores.std()
        ),
        "min: %.1f" % test_scores.min(),
        "max: %.1f" % test_scores.max(),
    )


def run(game, agent, actions, num_epochs, frame_repeat, steps_per_epoch=2000):
    """Run training epochs with frame stacking"""
    start_time = time()
    
    for epoch in range(num_epochs):
        game.new_episode()
        agent.reset_hidden()
        train_scores = []
        global_step = 0
        print(f"\nEpoch #{epoch + 1}")
        
        # Initialize frame stack
        frame_stack = deque(maxlen=4)
        initial_state = preprocess(game.get_state().screen_buffer)
        for _ in range(4):
            frame_stack.append(initial_state)

        for _ in trange(steps_per_epoch, leave=False):
            # Stack frames for temporal information
            state = np.stack(frame_stack, axis=0)
            action = agent.get_action(state)
            reward = game.make_action(actions[action], frame_repeat)
            done = game.is_episode_finished()

            if not done:
                next_frame = preprocess(game.get_state().screen_buffer)
                frame_stack.append(next_frame)
                next_state = np.stack(frame_stack, axis=0)
            else:
                next_state = np.zeros((4, 30, 45)).astype(np.float32)

            agent.store_experience(state, action, reward, next_state, done)

            if global_step > agent.batch_size and global_step % 4 == 0:
                agent.train()

            if done:
                train_scores.append(game.get_total_reward())
                game.new_episode()
                agent.reset_hidden()
                # Reset frame stack
                if not game.is_episode_finished():
                    initial_state = preprocess(game.get_state().screen_buffer)
                    frame_stack.clear()
                    for _ in range(4):
                        frame_stack.append(initial_state)

            global_step += 1

        train_scores = np.array(train_scores)
        print(
            "Results: mean: {:.1f} +/- {:.1f},".format(
                train_scores.mean(), train_scores.std()
            ),
            "min: %.1f," % train_scores.min(),
            "max: %.1f," % train_scores.max(),
        )

        test(game, agent)
        if save_model:
            print("Saving the network weights to:", model_savefile)
            agent.save_model(model_savefile)
        print("Total elapsed time: %.2f minutes" % ((time() - start_time) / 60.0))

    game.close()
    return agent, game


class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration"""
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul(x.abs().sqrt())
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class ImprovedDQN(nn.Module):
    """Simplified but robust DQN architecture"""
    def __init__(self, action_size):
        super(ImprovedDQN, self).__init__()
        self.action_size = action_size
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate conv output size
        self.conv_output_size = self._get_conv_output_size()
        
        # Dueling architecture
        self.value_stream = nn.Sequential(
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )
        
    def _get_conv_output_size(self):
        # Test input to calculate output size
        x = torch.zeros(1, 4, 30, 45)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.view(1, -1).size(1)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(batch_size, -1)
        
        # Dueling streams
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine value and advantage
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer"""
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        
    def push(self, experience):
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
            
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
        
        return experiences, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6
            
    def __len__(self):
        return len(self.buffer)


class ImprovedDQNAgent:
    """Improved DQN Agent with essential features"""
    def __init__(
        self,
        action_size,
        memory_size=100000,
        batch_size=32,
        discount_factor=0.99,
        lr=0.00025,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        target_update_frequency=1000,
        load_model=False,
        model_path=None
    ):
        self.action_size = action_size
        self.batch_size = batch_size
        self.discount = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_frequency = target_update_frequency
        
        # Networks
        self.q_net = ImprovedDQN(action_size).to(DEVICE)
        self.target_net = ImprovedDQN(action_size).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        
        # Memory
        self.memory = PrioritizedReplayBuffer(memory_size)
        
        # Training variables
        self.steps = 0
        
        if load_model and model_path:
            self.load_model(model_path)
            
    def get_action(self, state, evaluate=False):
        """Select action using epsilon-greedy or greedy policy"""
        if evaluate or random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                q_values = self.q_net(state_tensor)
                return q_values.argmax().item()
        else:
            return random.randrange(self.action_size)
    
    def reset_hidden(self):
        """Compatibility method for LSTM-based models"""
        pass
        
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.memory.push(experience)
        
    def train(self):
        """Train the agent"""
        if len(self.memory) < self.batch_size:
            return
            
        # Sample from memory
        experiences, indices, weights = self.memory.sample(self.batch_size)
        
        # Prepare batch - convert to numpy arrays first for efficiency
        states_list = [e.state for e in experiences]
        actions_list = [e.action for e in experiences]
        rewards_list = [e.reward for e in experiences]
        next_states_list = [e.next_state for e in experiences]
        dones_list = [e.done for e in experiences]
        
        # Convert to tensors efficiently
        states = torch.FloatTensor(np.array(states_list)).to(DEVICE)
        actions = torch.LongTensor(actions_list).to(DEVICE)
        rewards = torch.FloatTensor(rewards_list).to(DEVICE)
        next_states = torch.FloatTensor(np.array(next_states_list)).to(DEVICE)
        dones = torch.BoolTensor(dones_list).to(DEVICE)
        
        # Current Q-values
        current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values (Double DQN)
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (self.discount * next_q_values * ~dones.unsqueeze(1))
        
        # Compute loss
        td_errors = target_q_values - current_q_values
        loss = (td_errors.pow(2) * weights.unsqueeze(1)).mean()
        
        # Update priorities
        priorities = td_errors.abs().detach().cpu().numpy().flatten()
        self.memory.update_priorities(indices, priorities)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
            
    def save_model(self, path):
        """Save model checkpoint"""
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
        
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=DEVICE)
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.steps = checkpoint.get('steps', 0)


if __name__ == "__main__":
    # Initialize game and actions
    game = create_simple_game()
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Initialize agent
    agent = ImprovedDQNAgent(
        action_size=len(actions),
        memory_size=replay_memory_size,
        batch_size=batch_size,
        discount_factor=discount_factor,
        lr=learning_rate,
        load_model=load_model
    )

    # Add method for compatibility
    def append_memory(state, action, reward, next_state, done):
        agent.store_experience(state, action, reward, next_state, done)
    
    agent.append_memory = append_memory

    # Run training
    if not skip_learning:
        agent, game = run(
            game,
            agent,
            actions,
            num_epochs=train_epochs,
            frame_repeat=frame_repeat,
            steps_per_epoch=learning_steps_per_epoch,
        )

        print("======================================")
        print("Training finished. It's time to watch!")

    # Reinitialize game for visualization
    game.close()
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.init()

    # Initialize frame stack for visualization
    frame_stack = deque(maxlen=4)

    for _ in range(episodes_to_watch):
        game.new_episode()
        agent.reset_hidden()
        
        # Initialize frame stack
        initial_state = preprocess(game.get_state().screen_buffer)
        frame_stack.clear()
        for _ in range(4):
            frame_stack.append(initial_state)
            
        while not game.is_episode_finished():
            state = np.stack(frame_stack, axis=0)
            best_action_index = agent.get_action(state, evaluate=True)

            # Smooth animation
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()
                
            # Update frame stack
            if not game.is_episode_finished():
                new_frame = preprocess(game.get_state().screen_buffer)
                frame_stack.append(new_frame)

        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)