import numpy as np
from collections import deque
import config
import os
import json
from datetime import datetime

class CurriculumManager:
    def __init__(self):
        self.current_level = 0
        self.performance_history = deque(maxlen=100)
        self.level_attempts = 0
        self.level_successes = 0
        self.min_episodes_per_level = 50
        self.success_threshold = 0.3  # 30% of success to advance to the next level
        self.save_dir = 'saves'
        
        # Definition of the curriculum - from easiest to most difficult
        self.curriculum_levels = [
            {
                'name': 'Basic Movement',
                'map': 'map01',
                'skill': 1,
                'timeout': 35 * 60 * 2,  # 2 minutes
                'objectives': ['survive', 'explore'],
                'min_reward': 1000,
                'description': 'Learn basic movements and survival'
            },
            {
                'name': 'Basic Combat',
                'map': 'map01', 
                'skill': 2,
                'timeout': 35 * 60 * 3,  # 3 minutes
                'objectives': ['kill_enemies', 'collect_items'],
                'min_reward': 5000,
                'description': 'Basic combat and item collection'
            },
            {
                'name': 'Item Collection',
                'map': 'map01',
                'skill': 3,
                'timeout': 35 * 60 * 4,  # 4 minutes
                'objectives': ['collect_all_items', 'find_secrets'],
                'min_reward': 10000,
                'description': 'Complete item collection and secret discovery'
            },
            {
                'name': 'Level Completion',
                'map': 'map01',
                'skill': 4,
                'timeout': 35 * 60 * 5,  # 5 minutes
                'objectives': ['complete_level'],
                'min_reward': 100000,
                'description': 'Complete the level'
            },
            {
                'name': 'Advanced Combat',
                'map': 'map02',
                'skill': 3,
                'timeout': 35 * 60 * 6,  # 6 minutes
                'objectives': ['survive', 'kill_enemies'],
                'min_reward': 15000,
                'description': 'Advanced combat on a new map'
            },
            {
                'name': 'Complex Navigation',
                'map': 'map02',
                'skill': 4,
                'timeout': 35 * 60 * 8,  # 8 minutes
                'objectives': ['complete_level', 'find_secrets'],
                'min_reward': 200000,
                'description': 'Complex navigation and level completion'
            },
            {
                'name': 'Multi-Level Mastery',
                'map': 'map03',
                'skill': 4,
                'timeout': 35 * 60 * 10,  # 10 minutes
                'objectives': ['complete_level', 'efficiency'],
                'min_reward': 500000,
                'description': 'Master multi-levels'
            }
        ]

    def get_current_level_config(self):
        """Return the current level configuration"""
        if self.current_level >= len(self.curriculum_levels):
            return self.curriculum_levels[-1]
        return self.curriculum_levels[self.current_level]

    def record_episode_result(self, episode_reward, episode_completed=False, episode_stats=None):
        """Record the result of an episode and determine if we can advance to the next level"""
        self.level_attempts += 1
        
        # Determine if it's a success based on the current level criteria
        level_config = self.get_current_level_config()
        is_success = self._evaluate_success(episode_reward, episode_completed, episode_stats, level_config)
        
        if is_success:
            self.level_successes += 1
        
        self.performance_history.append({
            'reward': float(episode_reward),
            'success': bool(is_success),
            'completed': bool(episode_completed),
            'level': int(self.current_level)
        })
        
        if self._should_advance_level():
            self._advance_level()
            return True  # Niveau avancé
        
        return False  # Pas d'avancement

    def _evaluate_success(self, reward, completed, stats, level_config):
        """Evaluate if an episode is considered a success"""
        objectives = level_config['objectives']
        min_reward = level_config['min_reward']
        
        if reward < min_reward:
            return False
        
        if 'complete_level' in objectives and not completed:
            return False
        
        if 'survive' in objectives and stats:
            if stats.get('survival_time', 0) < 30 * 35:
                return False
        
        if 'efficiency' in objectives and stats:
            if stats.get('completion_time', float('inf')) > level_config['timeout'] * 0.7:
                return False
        
        return True

    def _should_advance_level(self):
        """Determine if we should advance to the next level"""
        if self.level_attempts < self.min_episodes_per_level:
            return False
        
        if self.current_level >= len(self.curriculum_levels) - 1:
            return False
        
        success_rate = self.level_successes / self.level_attempts
        return success_rate >= self.success_threshold

    def _advance_level(self):
        """Advance to the next level of the curriculum"""
        if self.current_level < len(self.curriculum_levels) - 1:
            self.current_level += 1
            self.level_attempts = 0
            self.level_successes = 0
            print(f"\n🎯 CURRICULUM ADVANCEMENT!")
            print(f"📈 Advancing to level {self.current_level + 1}: {self.get_current_level_config()['name']}")
            print(f"📋 Objective: {self.get_current_level_config()['description']}")
            print("=" * 60)

    def get_adjusted_learning_rate(self, base_lr):
        """Adjust the learning rate according to the difficulty level"""
        level_factor = 1.0 - (self.current_level * 0.1)
        return base_lr * max(0.3, level_factor)

    def get_adjusted_exploration(self, base_epsilon):
        """Adjust the exploration according to the level"""
        if self.level_attempts < 20:
            return min(1.0, base_epsilon * 1.5)
        return base_epsilon

    def should_use_pretrained_weights(self):
        """Determine if we should use the pretrained weights of the previous level"""
        return self.current_level > 0 and self.level_attempts < 10

    def get_curriculum_stats(self):
        """Return the curriculum statistics"""
        recent_performance = list(self.performance_history)[-20:]
        recent_success_rate = sum(1 for ep in recent_performance if ep['success']) / len(recent_performance) if recent_performance else 0
        
        return {
            'current_level': self.current_level + 1,
            'level_name': self.get_current_level_config()['name'],
            'level_attempts': self.level_attempts,
            'level_successes': self.level_successes,
            'success_rate': self.level_successes / max(1, self.level_attempts),
            'recent_success_rate': recent_success_rate,
            'total_levels': len(self.curriculum_levels),
            'progress_percentage': (self.current_level / len(self.curriculum_levels)) * 100
        }

    def get_training_schedule(self):
        """Return the recommended training schedule"""
        level_config = self.get_current_level_config()
        stats = self.get_curriculum_stats()
        
        episodes_remaining = max(0, self.min_episodes_per_level - self.level_attempts)
        estimated_time = episodes_remaining * (level_config['timeout'] / (35 * 60))
        
        return {
            'episodes_remaining_min': episodes_remaining,
            'estimated_time_minutes': estimated_time,
            'next_evaluation': max(0, self.min_episodes_per_level - self.level_attempts),
            'recommendation': self._get_training_recommendation(stats)
        }

    def _get_training_recommendation(self, stats):
        """Generate a training recommendation"""
        success_rate = stats['success_rate']
        
        if success_rate < 0.1:
            return "Focus on basic survival and exploration. Consider adjusting hyperparameters."
        elif success_rate < 0.2:
            return "Good progress! Continue training with focus on primary objectives."
        elif success_rate < self.success_threshold:
            return "Close to advancement! Refine strategies to achieve objectives."
        else:
            return "Ready for next level advancement!"

    def save_curriculum_state(self, filepath):
        """Save the curriculum state"""
        def convert_to_serializable(obj):
            """Convert numpy objects to native Python types"""
            if isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalars
                return obj.item()
            elif isinstance(obj, (bool, int, float, str, type(None))):
                return obj
            else:
                return str(obj)
        
        state = {
            'current_level': int(self.current_level),
            'level_attempts': int(self.level_attempts),
            'level_successes': int(self.level_successes),
            'performance_history': convert_to_serializable(list(self.performance_history))
        }
        
        import json
        try:
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            print(f"[SUCCESS] Etat du curriculum sauvegarde dans {filepath}")
        except Exception as e:
            print(f"[ERROR] Erreur lors de la sauvegarde du curriculum: {e}")
            # Sauvegarde de secours sans formatting
            try:
                import pickle
                backup_path = str(filepath).replace('.json', '_backup.pkl')
                with open(backup_path, 'wb') as f:
                    pickle.dump(state, f)
                print(f"[BACKUP] Sauvegarde de secours cree: {backup_path}")
            except Exception as e2:
                print(f"[ERROR] Impossible de sauvegarder: {e2}")

    def load_curriculum_state(self, filepath):
        """Load the curriculum state"""
        try:
            import json
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.current_level = int(state['current_level'])
            self.level_attempts = int(state['level_attempts'])
            self.level_successes = int(state['level_successes'])
            self.performance_history = deque(state['performance_history'], maxlen=100)
            
            print(f"[SUCCESS] Curriculum state loaded from {filepath}")
            return True
            
        except FileNotFoundError:
            print("[INFO] No curriculum state found, starting from the beginning.")
            return False
        except json.JSONDecodeError as e:
            print(f"[WARN] Curriculum file corrupted ({e}), trying to load backup...")
            try:
                import pickle
                backup_path = str(filepath).replace('.json', '_backup.pkl')
                with open(backup_path, 'rb') as f:
                    state = pickle.load(f)
                
                self.current_level = int(state['current_level'])
                self.level_attempts = int(state['level_attempts']) 
                self.level_successes = int(state['level_successes'])
                self.performance_history = deque(state['performance_history'], maxlen=100)
                
                print("[SUCCESS] Curriculum state loaded from backup")
                return True
            except Exception as e2:
                print(f"[ERROR] Error loading backup: {e2}")
                return False
        except Exception as e:
            print(f"[ERROR] Error loading curriculum: {e}")
            return False