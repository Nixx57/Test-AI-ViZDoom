import os
import json
import time
import math
import numpy as np
from datetime import datetime
from collections import defaultdict, deque
import config

class AdvancedRewardSystem:
    def __init__(self, curriculum_manager=None):
        # Reference to the curriculum manager
        self.curriculum = curriculum_manager
        
        # Position tracking and exploration
        self.visited_positions = set()
        self.position_history = deque(maxlen=500)  # Increased for better analysis
        self.last_position = None
        self.movement_threshold = 10  # Minimum distance to consider movement
        self.last_angle = 0  # Last angle of view
        
        # Player state tracking
        self.last_health = 100
        self.last_armor = 0
        self.last_ammo = {}
        self.last_weapons = set()
        self.weapon_ammo = {}  # Weapon ammo tracking
        self.health_history = deque(maxlen=100)  # Health history for danger detection
        
        # Objective tracking
        self.last_killcount = 0
        self.last_itemcount = 0
        self.last_secretcount = 0
        self.episode_start_pos = None
        self.key_positions = []  # Positions where keys have been collected
        self.important_items = []  # Important items discovered
        self.danger_zones = defaultdict(int)  # Zones where the agent took damage
        self.stuck_counter = 0
        self.objective_positions = []  # Current objective positions
        self.last_objective_distance = float('inf')
        
        # Tactical behavior
        self.combat_mode = False
        self.last_enemy_seen = 0
        self.ammo_conservation = 1.0  # Ammo conservation factor
        
        # Episode statistics
        self.episode_stats = {
            'damage_taken': 0,
            'enemies_killed': 0,
            'items_collected': 0,
            'secrets_found': 0,
            'distance_traveled': 0,
            'time_survived': 0,
            'efficiency': 0.0,
            'accuracy': 0.0,
            'shots_fired': 0,
            'hits_landed': 0,
            'health_collected': 0,
            'armor_collected': 0,
            'ammo_collected': 0,
            'weapons_collected': 0,
            'secrets_found_list': [],
            'dangerous_encounters': 0,
            'successful_retreats': 0
        }
        
        # Reward configuration
        self.reward_config = {
            'exploration': 1.0,
            'combat': 1.0,
            'survival': 1.0,
            'objective': 1.0,
            'efficiency': 1.0
        }
        
    def reset_episode(self, initial_game_vars=None):
        """Reset episode variables"""
        # Save previous episode statistics
        if hasattr(self, 'episode_stats') and self.episode_stats.get('time_survived', 0) > 0:
            self._log_episode_stats()
        
        # Reset positions and orientation
        self.visited_positions.clear()
        self.position_history.clear()
        self.last_position = None
        self.last_angle = 0
        self.stuck_counter = 0
        
        # Reset player state
        self.last_health = initial_game_vars[7] if initial_game_vars and len(initial_game_vars) > 7 else 100
        self.last_armor = initial_game_vars[8] if initial_game_vars and len(initial_game_vars) > 8 else 0
        self.last_ammo.clear()
        self.last_weapons.clear()
        self.weapon_ammo.clear()
        self.health_history.clear()
        self.health_history.append(self.last_health)
        
        # Reset killcount
        self.last_killcount = initial_game_vars[0] if initial_game_vars and len(initial_game_vars) > 0 else 0
        
        # Reset episode statistics
        self.episode_stats = {
            'damage_taken': 0,
            'enemies_killed': 0,
            'items_collected': 0,
            'secrets_found': 0,
            'distance_traveled': 0,
            'time_survived': 0,
            'efficiency': 0.0,
            'accuracy': 0.0,
            'shots_fired': 0,
            'hits_landed': 0,
            'health_collected': 0,
            'armor_collected': 0,
            'ammo_collected': 0,
            'weapons_collected': 0,
            'secrets_found_list': [],
            'dangerous_encounters': 0,
            'successful_retreats': 0,
            'episode_start_time': time.time()
        }
        
        # Initialize ammo if game variables are provided
        if initial_game_vars and len(initial_game_vars) > 12:
            self.last_ammo = {
                'pistol': initial_game_vars[12],  # SELECTED_WEAPON_AMMO
                'shotgun': initial_game_vars[13],  # AMMO2
                'chaingun': initial_game_vars[14],  # AMMO3
                'rocket': initial_game_vars[15]    # AMMO4
            }
        
        # Reset objectives
        self.last_killcount = initial_game_vars[0] if initial_game_vars and len(initial_game_vars) > 0 else 0
        self.last_itemcount = initial_game_vars[1] if initial_game_vars and len(initial_game_vars) > 1 else 0
        self.last_secretcount = initial_game_vars[2] if initial_game_vars and len(initial_game_vars) > 2 else 0
        self.key_positions.clear()
        self.important_items.clear()
        self.danger_zones.clear()
        self.objective_positions = self._get_initial_objectives()
        self.last_objective_distance = float('inf')
        
        # Reset tactical behavior
        self.combat_mode = False
        self.last_enemy_seen = 0
        self.ammo_conservation = 1.0
        
        # Reset statistics
        self.episode_stats = {
            'damage_taken': 0,
            'enemies_killed': 0,
            'items_collected': 0,
            'secrets_found': 0,
            'distance_traveled': 0,
            'time_survived': 0,
            'efficiency': 0.0,
            'accuracy': 0.0,
            'shots_fired': 0,
            'hits_landed': 0,
            'health_collected': 0,
            'armor_collected': 0,
            'ammo_collected': 0,
            'weapons_collected': 0,
            'secrets_found_list': [],
            'dangerous_encounters': 0,
            'successful_retreats': 0,
            'episode_start_time': time.time()
        }

    def get_position_key(self, x, y, grid_size=32):
        """Convertit une position en clé de grille pour l'exploration"""
        return (int(x // grid_size), int(y // grid_size))

    def calculate_exploration_reward(self, current_pos):
        """Calcule la récompense d'exploration avec bonus pour la progression"""
        if current_pos is None:
            return 0.0
            
        if self.last_position is None:
            self.last_position = current_pos
            self.episode_start_pos = current_pos
            return 0.0
        
        # Calculate distance from the start position
        if not hasattr(self, 'episode_start_pos'):
            self.episode_start_pos = current_pos
            
        distance_from_start = math.sqrt(
            (current_pos[0] - self.episode_start_pos[0])**2 + 
            (current_pos[1] - self.episode_start_pos[1])**2
        )
        
        # Calculate distance traveled since the last frame
        distance = math.sqrt(
            (current_pos[0] - self.last_position[0])**2 + 
            (current_pos[1] - self.last_position[1])**2
        )
        
        # Update position
        self.last_position = current_pos
        self.episode_stats['distance_traveled'] += distance
        
        # Check if the agent is stuck
        if distance < self.movement_threshold:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            
        # Reward for discovering new zones
        pos_key = self.get_position_key(current_pos[0], current_pos[1])
        exploration_reward = 0.0
        
        if pos_key not in self.visited_positions:
            self.visited_positions.add(pos_key)
            exploration_reward = config.REWARD_EXPLORATION * 2.0  # Double the exploration reward
            
            # Bonus for moving away from the start position
            exploration_reward += min(1.0, distance_from_start / 100.0)
            
            # Bonus for exploring dangerous zones
            if pos_key in self.danger_zones:
                exploration_reward *= 1.5
        
        # Penalty for being stuck
        if self.stuck_counter > 100:  # ~3 seconds without significant movement
            exploration_reward -= 1.0
        
        # Bonus for global progress (moving away from the start position)
        exploration_reward += distance_from_start * 0.001  # Small linear bonus
            
        return exploration_reward

    def calculate_movement_reward(self, current_pos):
        """Reward/punishment for movement to avoid getting stuck or going in circles"""
        if current_pos is None or self.last_position is None:
            self.last_position = current_pos
            return 0
        
        # Calculate distance traveled
        distance = math.sqrt(
            (current_pos[0] - self.last_position[0])**2 + 
            (current_pos[1] - self.last_position[1])**2
        )
        
        # Add position to history
        self.position_history.append(current_pos)
        
        # Check for circular movement
        circular_movement_penalty = 0.0
        if len(self.position_history) > 20:  # About 0.5 seconds at 35 FPS
            # Calculate variance of recent positions
            recent_positions = list(self.position_history)[-20:]
            pos_array = np.array(recent_positions)
            
            # If variance is low, agent is circling
            if np.var(pos_array) < 0.1:  # Adjust threshold
                circular_movement_penalty = -0.1  # Pénalité pour mouvement circulaire
        
        # Update stuck counter
        if distance < self.movement_threshold:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        
        self.last_position = current_pos
        
        # Penalty if stuck for too long
        if self.stuck_counter > 30:  # ~1 second at 35 FPS
            return -5.0 + circular_movement_penalty
        
        return circular_movement_penalty  # Return penalty even if not stuck

    def calculate_health_reward(self, current_health, current_armor):
        """Reward for managing health and armor"""
        reward = 0
        
        # Reward for health pickup
        health_diff = current_health - self.last_health
        if health_diff > 0:
            reward += health_diff * config.REWARD_HEALTH_PICKUP
        elif health_diff < 0:
            # Penalty for taking damage
            reward += health_diff * 2  # More severe than gain
        
        # Reward for armor pickup
        armor_diff = current_armor - self.last_armor
        if armor_diff > 0:
            reward += armor_diff * config.REWARD_ARMOR_PICKUP
        
        self.last_health = current_health
        self.last_armor = current_armor
        
        return reward

    def calculate_combat_reward(self, game_vars):
        """Reward combat performance"""
        reward = 0
        current_killcount = game_vars[0]  # KILLCOUNT is the first
        
        # Reward for new kills (reduced to avoid exploitation)
        kill_diff = current_killcount - self.last_killcount
        if kill_diff > 0:
            # Kill bonus based on remaining health
            health_bonus = min(1.0, game_vars[7] / 50.0)  # HEALTH
            # Reduce base reward for easy kills
            reward += kill_diff * (config.REWARD_KILL * 0.5) * (1 + health_bonus)
        
        # Penalty for shooting without visible target
        if hasattr(self, 'last_shot_time'):
            time_since_last_shot = time.time() - self.last_shot_time
            if time_since_last_shot < 1.0:  # If agent shoots too frequently
                reward -= 0.05  # Light penalty for spamming
        
        # Update last_killcount is now handled in _update_episode_stats
        return reward

    def calculate_item_reward(self, game_vars):
        """Reward item collection"""
        reward = 0
        current_itemcount = game_vars[1]  # ITEMCOUNT
        current_secretcount = game_vars[2]  # SECRETCOUNT
        
        # Reward for new items
        item_diff = current_itemcount - self.last_itemcount
        if item_diff > 0:
            reward += item_diff * config.REWARD_ITEM_PICKUP
            self.episode_stats['items_collected'] += item_diff  # Update item counter
        
        # Important bonus for secrets
        secret_diff = current_secretcount - self.last_secretcount
        if secret_diff > 0:
            reward += secret_diff * config.REWARD_SECRET
            self.episode_stats['secrets_found'] += secret_diff  # Update secret counter
        
        self.last_itemcount = current_itemcount
        self.last_secretcount = current_secretcount
        
        return reward

    def calculate_weapon_management_reward(self, game_vars):
        """Reward intelligent weapon management"""
        reward = 0
        
        # Weapon variables (indices 16-25 in GAME_VARIABLES)
        current_weapons = set()
        for i in range(16, 26):  # WEAPON0 to WEAPON9
            if i < len(game_vars) and game_vars[i] > 0:
                current_weapons.add(i - 16)
        
        # Reward for new weapons
        new_weapons = current_weapons - self.last_weapons
        reward += len(new_weapons) * config.REWARD_WEAPON_PICKUP
        
        # Bonus for having multiple weapons (versatility)
        if len(current_weapons) >= 3:
            reward += 10
        
        self.last_weapons = current_weapons
        return reward
        
    def calculate_objective_reward(self, game_vars, current_pos, current_angle):
        """
        Calculate reward based on progress towards objectives
        
        Args:
            game_vars: Game variables
            current_pos: Current position (x, y, z)
            current_angle: Current view angle
            
        Returns:
            float: Reward for progress towards objectives
        """
        if not hasattr(self, 'episode_start_pos'):
            self.episode_start_pos = current_pos
            self.furthest_distance = 0
            self.last_objective_check = time.time()
            return 0.0
            
        # Calculate distance from the start position
        distance_from_start = math.sqrt(
            (current_pos[0] - self.episode_start_pos[0])**2 + 
            (current_pos[1] - self.episode_start_pos[1])**2
        )
        
        # Reward for breaking the distance record
        reward = 0.0
        if distance_from_start > self.furthest_distance + 10:  # 10 unit threshold
            reward += (distance_from_start - self.furthest_distance) * 0.01
            self.furthest_distance = distance_from_start
            
        # Periodic progress check
        current_time = time.time()
        if current_time - self.last_objective_check > 5.0:  # Every 5 seconds
            self.last_objective_check = current_time
            
            # Penalty if agent is not making progress
            if distance_from_start < 100:  # Still close to start
                reward -= 0.5
            elif distance_from_start < 200:  # Slow progress
                reward -= 0.2
                
        # Bonus for reaching distant areas
        if distance_from_start > 300:
            reward += 0.1
        elif distance_from_start > 500:
            reward += 0.2
            
        return reward

    def calculate_tactical_reward(self, game_vars, action_taken):
        """
        Calculate tactical rewards based on game context
        
        Args:
            game_vars: Game variables
            action_taken: Last action taken
            
        Returns:
            float: Tactical reward
        """
        if len(game_vars) < 13:  # Check if we have enough variables
            return 0.0
            
        reward = 0.0
        health = game_vars[7]  # HEALTH
        armor = game_vars[8]   # ARMOR
        selected_weapon = game_vars[11]  # SELECTED_WEAPON
        ammo = game_vars[12]   # SELECTED_WEAPON_AMMO
        
        # Detection of nearby enemies (approximation based on sounds or damage)
        enemy_nearby = False
        if len(self.health_history) >= 3:
            # Check if recent damage has been taken
            recent_damage = sum(1 for i in range(1, 4) if i < len(self.health_history) and 
                              self.health_history[-i] > self.health_history[-i-1])
            enemy_nearby = recent_damage > 0
        
        # Update combat mode
        if enemy_nearby:
            self.last_enemy_seen = 0
            if not self.combat_mode:
                self.combat_mode = True
                reward += 2.0  # Bonus for enemy detection
        else:
            self.last_enemy_seen += 1
            if self.last_enemy_seen > 30:  # ~1 second without seeing an enemy
                self.combat_mode = False
        
        # Tactical weapon and ammo management
        if self.combat_mode:
            # Reward for using the appropriate weapon
            weapon_score = self._evaluate_weapon_choice(selected_weapon, ammo, game_vars)
            reward += weapon_score
            
            # Penalty for shooting without visible target
            if action_taken and action_taken.get('ATTACK', 0) > 0 and not enemy_nearby:
                reward -= 0.5
        else:
            # In exploration mode, save ammo
            if action_taken and action_taken.get('ATTACK', 0) > 0:
                reward -= 1.0
        
        # Health and armor management
        if health < 30:
            # Reward for seeking health when it's low
            if action_taken and (action_taken.get('MOVE_FORWARD', 0) > 0 or 
                                action_taken.get('MOVE_BACKWARD', 0) > 0):
                reward += 0.2
            
            # Penalty for taking unnecessary risks
            if enemy_nearby and health < 15:
                reward -= 0.5
        
        # Update health history
        self.health_history.append(health)
        
        return reward
        
    def _evaluate_weapon_choice(self, weapon_id, ammo, game_vars):
        """Evaluate weapon choice based on context"""
        if weapon_id == 0:  # Fist
            return -1.0  # Avoid using fists if possible
            
        # Check ammo for this weapon
        weapon_ammo = self._get_weapon_ammo(weapon_id, game_vars)
        
        if weapon_ammo <= 0:
            return -2.0  # Strong penalty for empty weapon
            
        # Reward for using powerful weapons against strong enemies
        enemy_strength = self._estimate_enemy_strength(game_vars)
        
        if enemy_strength > 0.7 and weapon_id in [3, 4, 6, 7]:  # Powerful weapons
            return 1.5
        elif enemy_strength < 0.3 and weapon_id in [1, 2, 5]:  # Light weapons
            return 1.0
            
        return 0.0
        
    def _get_weapon_ammo(self, weapon_id, game_vars):
        """Get ammo for a given weapon"""
        # Mapping of weapons to their ammo types
        weapon_ammo_map = {
            1: 12,  # Pistol -> Clips
            2: 13,  # Shotgun -> Shells
            3: 14,  # Minigun -> Clips
            4: 15,  # RL -> Rocket
            5: 17,  # PR -> Cells
            6: 16,  # BFG -> Cells
            7: 15   # SSG -> Shells
        }
        
        ammo_index = weapon_ammo_map.get(weapon_id, -1)
        if ammo_index == -1 or ammo_index >= len(game_vars):
            return 0
            
        return game_vars[ammo_index]
        
    def _estimate_enemy_strength(self, game_vars):
        """Estimate the strength of enemies nearby"""
        # This method is an approximation based on recent damage
        if len(self.health_history) < 5:
            return 0.0
            
        # Calculate average damage over the last 5 frames
        damage_taken = 0
        for i in range(1, min(6, len(self.health_history))):
            damage = max(0, self.health_history[-i] - self.health_history[-(i+1)])
            damage_taken += damage
            
        # Normalize between 0 and 1
        return min(1.0, damage_taken / 50.0)
        
    def calculate_efficiency_reward(self, game_vars, action_taken):
        """
        Calculate efficiency reward based on action efficiency
        
        Args:
            game_vars: Game variables
            action_taken: Last action taken
            
        Returns:
            float: Efficiency reward
        """ 
        if not action_taken:
            return 0.0
            
        reward = 0.0
        
        # Reward for effective actions
        if action_taken.get('MOVE_FORWARD', 0) > 0 and not self.combat_mode:
            # Moving forward is good in exploration
            reward += 0.1
            
        # Penalty for unnecessary movements
        if (action_taken.get('TURN_LEFT', 0) > 0 and 
            action_taken.get('TURN_RIGHT', 0) > 0):
            # Turning left and right at the same time is inefficient
            reward -= 0.5
            
        # Reward for effective ammo usage
        if action_taken.get('ATTACK', 0) > 0 and self.combat_mode:
            # Bonus for shooting enemies
            reward += 0.2
            
            # Update ammo stats
            self.episode_stats['shots_fired'] += 1
            
            # Detection of hits (approximate)
            if len(self.health_history) >= 2 and self.health_history[-1] < self.health_history[-2]:
                self.episode_stats['hits_landed'] += 1
        
        return reward

    def calculate_objective_reward(self, game_vars, current_pos, current_angle):
        """
        Calculate objective reward based on progress towards objectives
        
        Args:
            game_vars: Game variables
            current_pos: Current position (x, y, z)
            current_angle: Current view angle
            
        Returns:
            float: Objective reward
        """
        if not self.objective_positions:
            return 0.0
            
        reward = 0.0
        
        # Kill objectives
        for obj in [o for o in self.objective_positions if o['type'] == 'kill']:
            if obj['target'] == 'enemies':
                kill_diff = game_vars[0] - self.last_killcount
                if kill_diff > 0:
                    reward += kill_diff * config.REWARD_KILL
                    self.episode_stats['enemies_killed'] += kill_diff
        
        # Find objectives
        for obj in [o for o in self.objective_positions if o['type'] == 'find']:
            if obj['target'] == 'secrets':
                secret_diff = game_vars[2] - self.last_secretcount
                if secret_diff > 0:
                    reward += secret_diff * config.REWARD_SECRET
                    self.episode_stats['secrets_found'] += secret_diff
        
        # Collect objectives
        for obj in [o for o in self.objective_positions if o['type'] == 'collect']:
            if obj['target'] == 'items':
                item_diff = game_vars[1] - self.last_itemcount
                if item_diff > 0:
                    reward += item_diff * config.REWARD_ITEM_PICKUP
                    self.episode_stats['items_collected'] += item_diff
        
        # Reward for getting closer to position objectives
        if len(current_pos) >= 2:  # Verify we have at least x and y
            for obj in [o for o in self.objective_positions if 'position' in o]:
                obj_pos = obj['position']
                distance = math.sqrt(
                    (current_pos[0] - obj_pos[0])**2 + 
                    (current_pos[1] - obj_pos[1])**2
                )
                
                # Reward for getting closer to the objective
                if distance < self.last_objective_distance:
                    reward += (self.last_objective_distance - distance) * 0.1
                    self.last_objective_distance = distance
                
                # Big bonus for having reached the objective
                if distance < 50:  # Distance threshold to consider objective reached
                    reward += config.REWARD_OBJECTIVE_REACHED
                    self.objective_positions.remove(obj)  # Remove the reached objective
        
        return reward

    def get_total_reward(self, base_reward, game_vars, action_taken=None, done=False, game_state=None):
        """
        Calculate total reward with all modifiers
        
        Args:
            base_reward: Base reward from the environment
            game_vars: Game variables (health, armor, etc.)
            action_taken: Last action taken
            done: If the episode is done
            game_state: Complete game state (optional)
            
        Returns:
            float: Total reward calculated
        """
        if len(game_vars) < len(config.GAME_VARIABLES):
            return base_reward
        
        # Update survival time
        self.episode_stats['time_survived'] = time.time() - self.episode_stats.get('episode_start_time', time.time())
        
        # Retrieve current variables
        current_pos = (game_vars[3], game_vars[4], game_vars[5])  # POSITION_X, Y, Z
        current_health = game_vars[7]  # HEALTH
        current_armor = game_vars[8]  # ARMOR
        current_angle = game_vars[6]  # ANGLE
        
        # Update position history
        self.position_history.append(current_pos)
        
        # Calculate different components of the reward
        rewards = {
            'base': base_reward,
            'exploration': self.calculate_exploration_reward(current_pos),
            'movement': self.calculate_movement_reward(current_pos),
            'sante': self.calculate_health_reward(current_health, current_armor),
            'combat': self.calculate_combat_reward(game_vars),
            'objets': self.calculate_item_reward(game_vars),
            'armes': self.calculate_weapon_management_reward(game_vars),
            'tactique': self.calculate_tactical_reward(game_vars, action_taken),
            'objectifs': self.calculate_objective_reward(game_vars, current_pos, current_angle),
            'efficacite': self.calculate_efficiency_reward(game_vars, action_taken)
        }
        
        # Apply configuration weights
        weighted_rewards = {
            'base': rewards['base'],
            'exploration': rewards['exploration'] * self.reward_config['exploration'],
            'combat': rewards['combat'] * self.reward_config['combat'],
            'sante': rewards['sante'] * self.reward_config['survival'],
            'objectifs': rewards['objectifs'] * self.reward_config['objective'],
            'efficacite': rewards['efficacite'] * self.reward_config['efficiency']
        }
        
        # Calculate total reward
        total_reward = sum(weighted_rewards.values())
        
        # Update statistics
        self._update_episode_stats(rewards, game_vars, done)
        
        # Handle episode end
        if done:
            self._handle_episode_end(base_reward > 0)
        
        return total_reward

    def _get_initial_objectives(self):
        """Determine initial objectives based on the current level"""
        if not self.curriculum:
            return []
            
        level_config = self.curriculum.get_current_level_config()
        objectives = []
        
        # Add objectives based on the level configuration
        if 'kill_enemies' in level_config.get('objectives', []):
            objectives.append({'type': 'kill', 'target': 'enemies', 'count': level_config.get('enemy_count', 0)})
            
        if 'find_secrets' in level_config.get('objectives', []):
            objectives.append({'type': 'find', 'target': 'secrets', 'count': level_config.get('required_secrets', 0)})
            
        if 'collect_items' in level_config.get('objectives', []):
            objectives.append({'type': 'collect', 'target': 'items', 'count': level_config.get('required_items', 0)})
            
        return objectives
        
    def _update_episode_stats(self, rewards, game_vars, done):
        """Update episode statistics"""
        # Update combat statistics
        self.episode_stats['damage_taken'] += max(0, self.last_health - game_vars[7])
        
        # Update accuracy
        if 'hits_landed' in self.episode_stats and 'shots_fired' in self.episode_stats:
            self.episode_stats['accuracy'] = (
                self.episode_stats['hits_landed'] / 
                max(1, self.episode_stats['shots_fired'])
            )
        
        # Update efficiency
        damage_dealt = game_vars[0] * 100  # Damage dealt based on enemies killed
        damage_taken = self.episode_stats['damage_taken']
        self.episode_stats['efficiency'] = (
            damage_dealt / max(1, damage_taken) if damage_taken > 0
            else damage_dealt
        )
        
        # Detect dangerous encounters
        if game_vars[7] < 30:  # Low health
            self.episode_stats['dangerous_encounters'] += 1
    
    def _handle_episode_end(self, success):
        """Handle episode end"""
        # Update final statistics
        self.episode_stats['success'] = success
        self.episode_stats['final_health'] = self.last_health
        self.episode_stats['final_armor'] = self.last_armor
        
        # Log episode statistics
        self._log_episode_stats()
    
    def _log_episode_stats(self):
        """ Log episode statistics"""
        if not hasattr(self, 'episode_stats') or 'time_survived' not in self.episode_stats:
            return
            
        stats = self.episode_stats.copy()
        stats['timestamp'] = datetime.now().isoformat()
        
        # Create log directory if necessary
        log_dir = os.path.join('logs', 'episodes')
        os.makedirs(log_dir, exist_ok=True)
        
        # File name based on date and time
        filename = f"episode_{int(time.time())}.json"
        filepath = os.path.join(log_dir, filename)
        
        # Write statistics to a JSON file
        try:
            with open(filepath, 'w') as f:
                json.dump(stats, f, indent=2)
        except Exception as e:
            print(f"Error logging episode statistics: {e}")
    
    def get_detailed_stats(self):
        """Return detailed statistics for the current episode"""
        return self.episode_stats
        
    def get_exploration_stats(self):
        """
        Return exploration statistics
        
        Returns:
            dict: Dictionary containing exploration statistics
        """
        return {
            'visited_positions': len(self.visited_positions),
            'position_history_length': len(self.position_history),
            'danger_zones_count': len(self.danger_zones),
            'important_items': len(self.important_items),
            'key_positions': len(self.key_positions)
        }

    def _update_episode_stats(self, rewards, game_vars, done):
        """Update episode statistics"""
        # Update combat statistics
        current_health = game_vars[7]  # HEALTH
        damage_taken = max(0, self.last_health - current_health)
        self.episode_stats['damage_taken'] += damage_taken
        
        # Update number of enemies killed
        current_kills = game_vars[0]  # KILLCOUNT
        if hasattr(self, 'last_killcount'):
            kill_diff = current_kills - self.last_killcount
            if kill_diff > 0:
                self.episode_stats['enemies_killed'] += kill_diff
        
        # Update accuracy
        if 'hits_landed' in self.episode_stats and 'shots_fired' in self.episode_stats:
            self.episode_stats['accuracy'] = (
                self.episode_stats['hits_landed'] / 
                max(1, self.episode_stats['shots_fired'])
            )
        
        # Update efficiency
        damage_dealt = game_vars[0] * 100  # Damage dealt based on enemies killed
        damage_taken = self.episode_stats['damage_taken']
        self.episode_stats['efficiency'] = (
            damage_dealt / max(1, damage_taken) if damage_taken > 0
            else damage_dealt
        )
        
        # Detect dangerous encounters
        if current_health < 30:  # Low health
            self.episode_stats['dangerous_encounters'] += 1
        
        # Update references for the next tick
        self.last_health = current_health
        self.last_killcount = current_kills

    def _handle_episode_end(self, success):
        """Handle episode end"""
        # Update final statistics
        self.episode_stats['success'] = success
        self.episode_stats['final_health'] = self.last_health
        self.episode_stats['final_armor'] = self.last_armor
        
        # Log episode statistics
        self._log_episode_stats()

    def _log_episode_stats(self):
        """Log episode statistics"""
        if not hasattr(self, 'episode_stats') or 'time_survived' not in self.episode_stats:
            return
            
        stats = self.episode_stats.copy()
        stats['timestamp'] = datetime.now().isoformat()
        
        # Create log directory if necessary
        log_dir = os.path.join('logs', 'episodes')
        os.makedirs(log_dir, exist_ok=True)
        
        # File name based on date and time
        filename = f"episode_{int(time.time())}.json"
        filepath = os.path.join(log_dir, filename)
        
        # Write statistics to a JSON file
        try:
            with open(filepath, 'w') as f:
                json.dump(stats, f, indent=2)
        except Exception as e:
            print(f"Error logging episode statistics: {e}")

    def get_detailed_stats(self):
        """Return detailed statistics for the current episode"""
        return self.episode_stats

    def get_exploration_stats(self):
        """
        Return exploration statistics
        
        Returns:
            dict: Dictionary containing exploration statistics
        """
        return {
            'visited_positions': len(self.visited_positions),
            'position_history_length': len(self.position_history),
            'danger_zones_count': len(self.danger_zones),
            'important_items': len(self.important_items),
            'key_positions': len(self.key_positions)
        }