import os
import random
import time
from pathlib import Path
import pickle
import vizdoom as vzd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import cv2

import config
import utils
from reward_system import AdvancedRewardSystem
from curriculum_learning import CurriculumManager

class PPOActorCritic(nn.Module):
    def __init__(self, num_actions, input_shape=(1, 50, 80), num_game_variables=len(config.GAME_VARIABLES)):
        super(PPOActorCritic, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )

        self.fc_input_dim = self._get_conv_output(shape=input_shape)
        
        self.game_var_processor = nn.Sequential(
            nn.Linear(num_game_variables, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.lstm_input_dim = self.fc_input_dim + 64

        self.lstm = nn.LSTM(self.lstm_input_dim, 512, num_layers=2, batch_first=True, dropout=0.1)

        self.actor_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_actions)
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

        self.hidden = None

    def _get_conv_output(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def init_hidden(self, batch_size=1):
        device = next(self.parameters()).device
        h = torch.zeros(2, batch_size, 512).to(device)  # 2 layers
        c = torch.zeros(2, batch_size, 512).to(device)
        self.hidden = (h, c)

    def forward(self, x, game_vars):
        batch_size = x.size(0)
        
        # Image processing
        conv_out = self.conv(x).view(batch_size, -1)
        
        # Game variables processing
        game_processed = self.game_var_processor(game_vars)
        
        combined_input = torch.cat((conv_out, game_processed), dim=1).unsqueeze(1)

        lstm_out, self.hidden = self.lstm(combined_input, self.hidden)
        lstm_out = lstm_out.squeeze(1)

        action_logits = self.actor_head(lstm_out)
        state_value = self.critic_head(lstm_out)
        
        return action_logits, state_value

class PPOAgent:
    def __init__(self, num_actions, input_shape, num_game_variables, lr=0.00025, gamma=0.99, 
                 gae_lambda=0.95, clip_epsilon=0.2, ppo_epochs=10, mini_batch_size=64):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PPOActorCritic(num_actions, input_shape, num_game_variables).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        
        # Adjust learning rate
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)

        self.states = []
        self.game_vars = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def remember(self, state, game_var, action, log_prob, reward, done, value):
        self.states.append(state)
        self.game_vars.append(game_var)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def choose_action(self, state, game_var, exploration_bonus=0.0):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device) / 255.0
        game_var = torch.FloatTensor(game_var).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_logits, value = self.model(state, game_var)

        if exploration_bonus > 0:
            noise = torch.randn_like(action_logits) * exploration_bonus
            action_logits = action_logits + noise

        dist = Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()

    def learn(self):
        if len(self.states) < self.mini_batch_size:
            print(f"Warning: Not enough samples ({len(self.states)}) for training. Skipping update.")
            self.clear_memory()
            return None

        if len(self.rewards) != len(self.values) - 1:
            print(f"Error: Inconsistent data sizes. Rewards: {len(self.rewards)}, Values: {len(self.values)}")
            self.clear_memory()
            return None

        np_rewards = np.array(self.rewards)
        np_dones = np.array(self.dones)
        np_values = np.array(self.values)

        states = torch.FloatTensor(np.array(self.states[:-1])).to(self.device) / 255.0  # Remove last state
        game_vars = torch.FloatTensor(np.array(self.game_vars[:-1])).to(self.device)   # Remove last game_var
        actions = torch.LongTensor(self.actions[:-1]).to(self.device)                  # Remove last action
        log_probs_old = torch.FloatTensor(self.log_probs[:-1]).to(self.device)        # Remove last log_prob
        values_old = torch.FloatTensor(np_values[:-1]).to(self.device)                # Values corresponding to states
        next_value = np_values[-1]  # Last value for bootstrapping

        # GAE
        advantages = np.zeros(len(np_rewards), dtype=np.float32)
        last_advantage = 0
        for t in reversed(range(len(np_rewards))):
            if t == len(np_rewards) - 1:
                delta = np_rewards[t] + self.gamma * next_value * (1 - int(np_dones[t])) - np_values[t]
            else:
                delta = np_rewards[t] + self.gamma * np_values[t+1] * (1 - int(np_dones[t])) - np_values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - int(np_dones[t])) * last_advantage
            last_advantage = advantages[t]

        advantages = torch.FloatTensor(advantages).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values_old

        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0

        for epoch in range(self.ppo_epochs):
            batch_size = min(self.mini_batch_size, states.shape[0])
            num_batches = states.shape[0] // batch_size
            indices = np.arange(states.shape[0])
            np.random.shuffle(indices)

            for i in range(num_batches):
                batch_indices = indices[i * batch_size:(i + 1) * batch_size]

                batch_states = states[batch_indices]
                batch_game_vars = game_vars[batch_indices]
                batch_actions = actions[batch_indices]
                batch_log_probs_old = log_probs_old[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                self.model.init_hidden(batch_size=batch_size)
                action_logits, values = self.model(batch_states, batch_game_vars)
                dist = Categorical(logits=action_logits)
                log_probs = dist.log_prob(batch_actions)
                values = values.squeeze(1)

                ratio = torch.exp(log_probs - batch_log_probs_old)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * (batch_returns - values).pow(2).mean()
                entropy = dist.entropy().mean()
                
                entropy_coeff = 0.01 + 0.05 * max(0, 1 - epoch / self.ppo_epochs)
                loss = actor_loss + 0.5 * critic_loss - entropy_coeff * entropy

                self.optimizer.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()

        self.scheduler.step()
        
        self.clear_memory()
        
        return {
            'actor_loss': total_actor_loss / (self.ppo_epochs * num_batches),
            'critic_loss': total_critic_loss / (self.ppo_epochs * num_batches),
            'entropy': total_entropy / (self.ppo_epochs * num_batches),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }

    def clear_memory(self):
        self.states = []
        self.game_vars = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

def preprocess_screen(screen_buffer):
    """Préprocessing écran"""
    gray = cv2.cvtColor(screen_buffer, cv2.COLOR_RGB2GRAY)
    
    resized = cv2.resize(gray, (80, 50), interpolation=cv2.INTER_LINEAR)
    
    resized = cv2.equalizeHist(resized)
    
    processed = np.expand_dims(resized, axis=0)
    
    return processed.astype(np.float32)

def setup_game(curriculum_manager):
    game = vzd.DoomGame()
    
    level_config = curriculum_manager.get_current_level_config()
    
    game.set_doom_map(level_config['map'])
    game.set_doom_skill(level_config['skill'])
    game.set_episode_timeout(level_config['timeout'])
    
    game.set_screen_resolution(config.SCREEN_RESOLUTION)
    game.set_screen_format(config.SCREEN_FORMAT)
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.set_automap_buffer_enabled(True)
    game.set_objects_info_enabled(True)
    game.set_sectors_info_enabled(True)
    game.set_render_hud(True)
    game.set_render_minimal_hud(False)
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)
    game.set_render_particles(True)
    game.set_render_effects_sprites(True)
    game.set_render_messages(True)
    game.set_render_corpses(True)
    game.set_render_screen_flashes(True)
    game.set_available_buttons(config.AVAILABLE_BUTTONS)
    game.set_available_game_variables(config.GAME_VARIABLES)
    game.set_window_visible(config.WINDOW_VISIBLE)
    game.set_sound_enabled(config.SOUND)
    game.set_living_reward(config.REWARD_LIVING)
    game.set_death_reward(config.REWARD_DEATH)
    game.set_mode(config.VIZDOOM_MODE)
    
    game.init()
    return game

def main():
    """Main training function"""
    print("[GAME] Starting ViZDoom training with curriculum learning")
    print("=" * 60)
    
    # Init
    curriculum_manager = CurriculumManager()
    reward_system = AdvancedRewardSystem()
    
    curriculum_state_path = config.MODEL_SAVE_PATH / "curriculum_state.json"
    curriculum_manager.load_curriculum_state(curriculum_state_path)
    
    game = setup_game(curriculum_manager)
    
    # Init agent
    num_actions = len(config.ACTIONS_LIST)
    input_shape = (1, 50, 80)
    num_game_variables = len(config.GAME_VARIABLES)
    
    agent = PPOAgent(num_actions, input_shape, num_game_variables)
    
    # Episode tracking
    start_episode = 0
    episode_count = 0
    episode_rewards = []
    
    # Load existing checkpoint if available
    checkpoint_path = config.MODEL_SAVE_PATH / "ppo_doom_model.pth"
    if checkpoint_path.exists():
        print("[LOADING] Loading existing model...")
        try:
            # Try loading with weights_only=True (safer)
            checkpoint = torch.load(checkpoint_path, map_location=agent.device, weights_only=True)
        except (pickle.UnpicklingError, RuntimeError) as e:
            print("  - Loading with weights_only=True failed, trying with weights_only=False...")
            # If that fails, try with weights_only=False (less safe)
            checkpoint = torch.load(checkpoint_path, map_location=agent.device, weights_only=False)
            
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_episode = checkpoint.get('episode', 0) + 1
        episode_rewards = checkpoint.get('total_rewards', [])
        print(f"[SUCCESS] Model loaded - Resuming at episode {start_episode}")
    
    # Display curriculum statistics
    curriculum_stats = curriculum_manager.get_curriculum_stats()
    print("\n[STATE] Curriculum state:")
    print(f"   Current level: {curriculum_stats['level_name']} ({curriculum_stats['current_level']}/{curriculum_stats['total_levels']})")
    print(f"   Progress: {curriculum_stats['progress_percentage']:.1f}%")
    print(f"   Success rate: {curriculum_stats['success_rate']:.2%}")
    
    last_save_time = time.time()
    save_interval = 300  # Save every 5 minutes
    
    try:
        while episode_count < config.EPISODE_COUNT:
            episode_start_time = time.time()
            
            game.new_episode()
            agent.model.init_hidden()
            reward_system.reset_episode()
            
            current_episode_reward = 0
            step_count = 0
            episode_stats = {
                'survival_time': 0,
                'completion_time': 0,
                'kills': 0,
                'items_collected': 0,
                'secrets_found': 0
            }
            
            state = game.get_state()
            
            while not game.is_episode_finished():
                screen_buffer = state.screen_buffer
                game_variables = state.game_variables
                processed_screen = preprocess_screen(screen_buffer)
                
                # Optional
                if config.WINDOW_VISIBLE:
                    display_frame = cv2.resize(processed_screen[0], (320, 200))
                    cv2.imshow("Vision IA", display_frame.astype(np.uint8))
                    cv2.waitKey(1)
                
                # Adaptive exploration
                exploration_bonus = curriculum_manager.get_adjusted_exploration(0.1)
                
                # action
                action_idx, log_prob, value = agent.choose_action(
                    processed_screen, game_variables, exploration_bonus
                )
                action = config.ACTIONS_LIST[action_idx]
                
                base_reward = game.make_action(action)
                done = game.is_episode_finished()
                
                # Convert action index to action dictionary
                action_vector = config.ACTIONS_LIST[action_idx]
                action_dict = {}
                
                # Create dictionary with button names as keys
                for i, button in enumerate(config.AVAILABLE_BUTTONS):
                    button_name = str(button).split('.')[-1]  # Get just the button name
                    action_dict[button_name] = bool(action_vector[i])
                
                enhanced_reward = reward_system.get_total_reward(
                    base_reward, game_variables, action_dict, done
                )
                
                # Experience memory
                agent.remember(processed_screen, game_variables, action_idx, 
                             log_prob, enhanced_reward, done, value)
                
                current_episode_reward += enhanced_reward
                step_count += 1
                
                # Update stats
                episode_stats['survival_time'] = step_count
                if len(game_variables) > 0:
                    episode_stats['kills'] = game_variables[0]  # KILLCOUNT
                if len(game_variables) > 1:
                    episode_stats['items_collected'] = game_variables[1]  # ITEMCOUNT
                if len(game_variables) > 2:
                    episode_stats['secrets_found'] = game_variables[2]  # SECRETCOUNT
                
                if not done:
                    state = game.get_state()
                else:
                    state = None 
            
            episode_end_time = time.time()
            episode_stats['completion_time'] = episode_end_time - episode_start_time
            
            if done:
                agent.values.append(0.0)
            else:
                if state is not None:
                    final_screen = preprocess_screen(state.screen_buffer)
                    final_game_vars = state.game_variables
                    _, _, final_value = agent.choose_action(final_screen, final_game_vars, 0.0)
                    agent.values.append(final_value)
                else:
                    agent.values.append(0.0)
            
            episode_rewards.append(current_episode_reward)
            
            training_stats = agent.learn()
            
            episode_completed = current_episode_reward > 0
            level_advanced = curriculum_manager.record_episode_result(
                current_episode_reward, episode_completed, episode_stats
            )
            
            if level_advanced:
                game.close()
                game = setup_game(curriculum_manager)
            
            if episode_count % 10 == 0 or level_advanced:
                curriculum_stats = curriculum_manager.get_curriculum_stats()
                exploration_stats = reward_system.get_exploration_stats()
                
                print(f"\n[EPISODE] Episode #{start_episode + episode_count}")
                print(f"Level: {curriculum_stats['level_name']}")
                print(f"Score: {current_episode_reward:.2f} (Average over 100: {np.mean(episode_rewards[-100:]):.2f})")
                print(f"Best score: {max(episode_rewards):.2f}")
                print(f"Success rate: {curriculum_stats['success_rate']:.2%}")
                print(f"Visited positions: {exploration_stats['visited_positions']}")
                if training_stats is not None:
                    print(f"   LR: {training_stats['learning_rate']:.6f}")
                    print(f"   Entropy: {training_stats['entropy']:.4f}")
                    print(f"   Learning: Not enough data for this update")
            
            current_time = time.time()
            if current_time - last_save_time > save_interval:
                print("[SAUVEGARDE] Automatic save...")
                torch.save({
                    'episode': start_episode + episode_count,
                    'model_state_dict': agent.model.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'total_rewards': episode_rewards,
                    'curriculum_stats': curriculum_manager.get_curriculum_stats()
                }, checkpoint_path)
                
                curriculum_manager.save_curriculum_state(curriculum_state_path)
                last_save_time = current_time
            
            episode_count += 1
    
    except KeyboardInterrupt:
        print("\n[STOP] Stop requested by user")
    
    finally:
        print("[SAUVEGARDE] Final save...")
        torch.save({
            'episode': start_episode + episode_count,
            'model_state_dict': agent.model.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'total_rewards': episode_rewards,
            'curriculum_stats': curriculum_manager.get_curriculum_stats()
        }, checkpoint_path)
        
        curriculum_manager.save_curriculum_state(curriculum_state_path)
        
        final_stats = curriculum_manager.get_curriculum_stats()
        print(f"\n[TERMINE] Training completed!")
        print(f"   Episodes completed: {episode_count}")
        print(f"   Level reached: {final_stats['level_name']}")
        print(f"   Curriculum progress: {final_stats['progress_percentage']:.1f}%")
        
        game.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()