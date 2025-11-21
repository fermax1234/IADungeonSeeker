import gymnasium as gym
import numpy as np
import pygame
import random
from collections import defaultdict
import time
import matplotlib.pyplot as plt

class DungeonSeekersEnv(gym.Env):
    def __init__(self, size=10):
        super().__init__()
        self.size = size
        self.action_space = gym.spaces.Discrete(4)  # 0: up, 1: right, 2: down, 3: left
        # Expand observation space to include maze information
        self.observation_space = gym.spaces.Box(low=0, high=size-1, shape=(6 + size*size,), dtype=np.int32)
        
        # Initialize positions
        self.reset()
        
        # Visualization
        pygame.init()
        self.cell_size = 60
        self.screen = pygame.display.set_mode((self.size * self.cell_size, self.size * self.cell_size))
        pygame.display.set_caption("Dungeon Seekers - Multi-Agent RL")
        self.font = pygame.font.SysFont('Arial', 20)
        
        # Colors
        self.colors = {
            'empty': (255, 255, 255),
            'wall': (100, 100, 100),
            'seeker': (0, 0, 255),
            'hunter': (255, 0, 0),
            'key': (255, 215, 0),
            'start': (0, 255, 0),
            'text': (0, 0, 0)
        }
        
    def reset(self):
        # Create maze with obstacles using a more structured approach
        self.maze = np.zeros((self.size, self.size))
        
        # Add perimeter walls
        self.maze[0, :] = 1
        self.maze[-1, :] = 1
        self.maze[:, 0] = 1
        self.maze[:, -1] = 1
        
        # Add strategic obstacles (15% of cells)
        obstacle_count = int(self.size * self.size * 0.15)
        for _ in range(obstacle_count):
            x, y = random.randint(1, self.size-2), random.randint(1, self.size-2)
            # Ensure obstacles don't block critical paths completely
            if not ((x == self.size//2 and y == self.size//2) or 
                    (x <= 2 and y <= 2) or 
                    (x >= self.size-3 and y >= self.size-3)):
                self.maze[y, x] = 1
        
        # Initialize agent positions
        self.seeker_pos = np.array([1, 1])
        self.hunter_pos = np.array([self.size-2, self.size-2])
        self.key_pos = np.array([self.size//2, self.size//2])
        self.key_found = False
        self.seeker_has_key = False
        self.steps = 0
        self.max_steps = 200
        
        return self._get_observation()
    
    def _get_observation(self):
        # Return enhanced observation including maze structure
        basic_obs = np.array([
            self.seeker_pos[0], self.seeker_pos[1],
            self.hunter_pos[0], self.hunter_pos[1],
            self.key_pos[0], self.key_pos[1],
            int(self.seeker_has_key)
        ])
        
        # Flatten maze and combine with basic observation
        maze_flat = self.maze.flatten()
        full_obs = np.concatenate([basic_obs, maze_flat])
        return full_obs
    
    def step(self, seeker_action, hunter_action):
        self.steps += 1
        reward_seeker = -0.1  # Reduced step penalty
        reward_hunter = -0.1  # Reduced step penalty
        done = False
        info = {}
        
        # Move seeker with collision detection
        new_seeker_pos = self._move_agent(self.seeker_pos, seeker_action)
        if self._is_valid_position(new_seeker_pos):
            self.seeker_pos = new_seeker_pos
        else:
            reward_seeker -= 0.5  # Small penalty for hitting walls
        
        # Move hunter with collision detection
        new_hunter_pos = self._move_agent(self.hunter_pos, hunter_action)
        if self._is_valid_position(new_hunter_pos):
            self.hunter_pos = new_hunter_pos
        else:
            reward_hunter -= 0.5  # Small penalty for hitting walls
        
        # Distance-based shaping rewards
        if not self.seeker_has_key:
            # Seeker gets reward for getting closer to key
            dist_to_key = np.linalg.norm(self.seeker_pos - self.key_pos)
            reward_seeker += 0.1 * (1 / (dist_to_key + 1))  # Normalized distance reward
            
            # Hunter gets reward for getting closer to seeker
            dist_to_seeker = np.linalg.norm(self.hunter_pos - self.seeker_pos)
            reward_hunter += 0.1 * (1 / (dist_to_seeker + 1))
        else:
            # Seeker gets reward for getting closer to start
            dist_to_start = np.linalg.norm(self.seeker_pos - np.array([1, 1]))
            reward_seeker += 0.1 * (1 / (dist_to_start + 1))
            
            # Hunter gets higher reward for getting closer to seeker with key
            dist_to_seeker = np.linalg.norm(self.hunter_pos - self.seeker_pos)
            reward_hunter += 0.2 * (1 / (dist_to_seeker + 1))
        
        # Check if seeker found the key
        if not self.seeker_has_key and np.array_equal(self.seeker_pos, self.key_pos):
            self.seeker_has_key = True
            reward_seeker += 20  # Increased reward for finding key
            info['key_found'] = True
            print("KEY FOUND!")
        
        # Check if seeker returned to start with key
        if self.seeker_has_key and np.array_equal(self.seeker_pos, [1, 1]):
            reward_seeker += 30  # Increased reward for completing mission
            done = True
            info['mission_complete'] = True
            print("MISSION COMPLETE! Seeker won!")
        
        # Check if hunter caught seeker
        if np.array_equal(self.seeker_pos, self.hunter_pos):
            reward_hunter += 30  # Increased reward for catching seeker
            reward_seeker -= 10  # Increased penalty for being caught
            done = True
            info['seeker_caught'] = True
            print("SEEKER CAUGHT! Hunter won!")
        
        # Check if maximum steps reached
        if self.steps >= self.max_steps:
            done = True
            info['max_steps_reached'] = True
            
            # Penalize both agents for timeout
            if self.seeker_has_key:
                reward_seeker -= 5  # Seeker failed to return
                reward_hunter -= 10  # Hunter failed to catch seeker with key
            else:
                reward_seeker -= 10  # Seeker failed to find key
                reward_hunter -= 5   # Hunter failed to catch seeker
        
        return self._get_observation(), (reward_seeker, reward_hunter), done, info
    
    def _move_agent(self, position, action):
        new_pos = position.copy()
        if action == 0:  # Up
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == 1:  # Right
            new_pos[0] = min(self.size - 1, new_pos[0] + 1)
        elif action == 2:  # Down
            new_pos[1] = min(self.size - 1, new_pos[1] + 1)
        elif action == 3:  # Left
            new_pos[0] = max(0, new_pos[0] - 1)
        return new_pos
    
    def _is_valid_position(self, pos):
        x, y = pos
        # Check if position is within bounds and not a wall
        return 0 <= x < self.size and 0 <= y < self.size and self.maze[y, x] == 0
    
    def render(self):
        self.screen.fill(self.colors['empty'])
        
        # Draw walls
        for y in range(self.size):
            for x in range(self.size):
                if self.maze[y, x] == 1:
                    pygame.draw.rect(self.screen, self.colors['wall'], 
                                    (x * self.cell_size, y * self.cell_size, 
                                     self.cell_size, self.cell_size))
        
        # Draw key
        pygame.draw.rect(self.screen, self.colors['key'], 
                        (self.key_pos[0] * self.cell_size, self.key_pos[1] * self.cell_size, 
                         self.cell_size, self.cell_size))
        
        # Draw start position
        pygame.draw.rect(self.screen, self.colors['start'], 
                        (1 * self.cell_size, 1 * self.cell_size, self.cell_size, self.cell_size), 2)
        
        # Draw seeker
        seeker_color = self.colors['seeker']
        if self.seeker_has_key:
            seeker_color = (0, 255, 255)  # Cyan when carrying key
        pygame.draw.circle(self.screen, seeker_color, 
                          (int((self.seeker_pos[0] + 0.5) * self.cell_size), 
                           int((self.seeker_pos[1] + 0.5) * self.cell_size)), 
                          self.cell_size // 3)
        
        # Draw hunter
        pygame.draw.circle(self.screen, self.colors['hunter'], 
                          (int((self.hunter_pos[0] + 0.5) * self.cell_size), 
                           int((self.hunter_pos[1] + 0.5) * self.cell_size)), 
                          self.cell_size // 3)
        
        # Display status information
        status_text = f"Steps: {self.steps}/{self.max_steps}"
        if self.seeker_has_key:
            status_text += " | KEY FOUND!"
        text_surface = self.font.render(status_text, True, self.colors['text'])
        self.screen.blit(text_surface, (10, 10))
        
        pygame.display.flip()
    
    def close(self):
        pygame.quit()

class ImprovedQLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, 
                 exploration_rate=1.0, exploration_decay=0.997, exploration_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        self.learning_rate_decay = 0.9995
        self.min_learning_rate = 0.01
    
    def act(self, state):
        state_key = tuple(state[:10])  # Use partial state for generalization
        if np.random.random() < self.exploration_rate:
            return np.random.randint(self.action_size)  # Explore
        return np.argmax(self.q_table[state_key])  # Exploit
    
    def learn(self, state, action, reward, next_state, done):
        state_key = tuple(state[:10])
        next_state_key = tuple(next_state[:10])
        
        current_q = self.q_table[state_key][action]
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state_key])
        
        # Update Q-value
        self.q_table[state_key][action] += self.learning_rate * (target_q - current_q)
        
        # Decay exploration rate and learning rate
        if done:
            self.exploration_rate = max(self.exploration_min, 
                                       self.exploration_rate * self.exploration_decay)
            self.learning_rate = max(self.min_learning_rate, 
                                   self.learning_rate * self.learning_rate_decay)

def train_agents(episodes=2000, render_every=200):
    env = DungeonSeekersEnv(size=10)
    
    # Initialize improved agents
    seeker_agent = ImprovedQLearningAgent(state_size=6 + env.size*env.size, action_size=4,
                                         learning_rate=0.2, exploration_decay=0.998)
    hunter_agent = ImprovedQLearningAgent(state_size=6 + env.size*env.size, action_size=4,
                                         learning_rate=0.2, exploration_decay=0.998)
    
    # Training metrics
    seeker_rewards = []
    hunter_rewards = []
    success_rates = []
    key_found_rates = []
    
    for episode in range(episodes):
        state = env.reset()
        total_seeker_reward = 0
        total_hunter_reward = 0
        done = False
        episode_info = {}
        key_found = False
        
        while not done:
            # Get actions from both agents
            seeker_action = seeker_agent.act(state)
            hunter_action = hunter_agent.act(state)
            
            # Take step in environment
            next_state, rewards, done, info = env.step(seeker_action, hunter_action)
            seeker_reward, hunter_reward = rewards
            
            # Update agents
            seeker_agent.learn(state, seeker_action, seeker_reward, next_state, done)
            hunter_agent.learn(state, hunter_action, hunter_reward, next_state, done)
            
            # Update metrics
            total_seeker_reward += seeker_reward
            total_hunter_reward += hunter_reward
            state = next_state
            
            # Store episode info
            episode_info = info
            if info.get('key_found', False):
                key_found = True
            
            # Render every N episodes for visualization
            if episode % render_every == 0:
                env.render()
                time.sleep(0.02)  # Faster visualization
        
        # Record metrics
        seeker_rewards.append(total_seeker_reward)
        hunter_rewards.append(total_hunter_reward)
        
        # Check if mission was successful
        mission_success = episode_info.get('mission_complete', False)
        success_rates.append(1 if mission_success else 0)
        key_found_rates.append(1 if key_found else 0)
        
        # Print progress
        if episode % 100 == 0:
            avg_seeker_reward = np.mean(seeker_rewards[-100:])
            avg_hunter_reward = np.mean(hunter_rewards[-100:])
            recent_success_rate = np.mean(success_rates[-100:]) * 100
            recent_key_rate = np.mean(key_found_rates[-100:]) * 100
            print(f"Episode {episode}:")
            print(f"  Seeker Avg Reward: {avg_seeker_reward:.2f}")
            print(f"  Hunter Avg Reward: {avg_hunter_reward:.2f}")
            print(f"  Success Rate: {recent_success_rate:.1f}%")
            print(f"  Key Found Rate: {recent_key_rate:.1f}%")
            print(f"  Seeker Exploration: {seeker_agent.exploration_rate:.3f}")
            print(f"  Hunter Exploration: {hunter_agent.exploration_rate:.3f}")
    
    env.close()
    
    # Plot comprehensive training progress
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(seeker_rewards, alpha=0.3, color='blue')
    plt.plot(pd.Series(seeker_rewards).rolling(50).mean(), color='blue')
    plt.title('Seeker Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(2, 3, 2)
    plt.plot(hunter_rewards, alpha=0.3, color='red')
    plt.plot(pd.Series(hunter_rewards).rolling(50).mean(), color='red')
    plt.title('Hunter Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(2, 3, 3)
    # Calculate moving average of success rate
    window = 50
    moving_avg = [np.mean(success_rates[i:i+window]) for i in range(len(success_rates)-window)]
    plt.plot(moving_avg)
    plt.title('Success Rate (Moving Average)')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.ylim(0, 1)
    
    plt.subplot(2, 3, 4)
    key_moving_avg = [np.mean(key_found_rates[i:i+window]) for i in range(len(key_found_rates)-window)]
    plt.plot(key_moving_avg)
    plt.title('Key Found Rate (Moving Average)')
    plt.xlabel('Episode')
    plt.ylabel('Key Found Rate')
    plt.ylim(0, 1)
    
    plt.subplot(2, 3, 5)
    plt.plot(seeker_rewards, alpha=0.1, color='blue', label='Seeker')
    plt.plot(hunter_rewards, alpha=0.1, color='red', label='Hunter')
    plt.plot(pd.Series(seeker_rewards).rolling(100).mean(), color='blue', linewidth=2)
    plt.plot(pd.Series(hunter_rewards).rolling(100).mean(), color='red', linewidth=2)
    plt.title('Both Agents Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return seeker_agent, hunter_agent

def demo_trained_agents(seeker_agent, hunter_agent, episodes=5):
    env = DungeonSeekersEnv(size=10)
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_seeker_reward = 0
        total_hunter_reward = 0
        
        print(f"\n--- Demo Episode {episode+1} ---")
        
        while not done:
            # Get actions from trained agents (no exploration)
            seeker_action = np.argmax(seeker_agent.q_table[tuple(state[:10])])
            hunter_action = np.argmax(hunter_agent.q_table[tuple(state[:10])])
            
            # Take step in environment
            next_state, rewards, done, info = env.step(seeker_action, hunter_action)
            seeker_reward, hunter_reward = rewards
            
            # Update metrics
            total_seeker_reward += seeker_reward
            total_hunter_reward += hunter_reward
            state = next_state
            
            # Render for visualization
            env.render()
            time.sleep(0.1)  # Slow down for better visualization
            
            # Check for key events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
        
        print(f"Seeker Total Reward: {total_seeker_reward}")
        print(f"Hunter Total Reward: {total_hunter_reward}")
        print(f"Episode Info: {info}")
    
    env.close()

# Add pandas for rolling averages
import pandas as pd

if __name__ == "__main__":
    print("Starting Improved Dungeon Seekers Training...")
    
    # Train the agents with improved parameters
    seeker_agent, hunter_agent = train_agents(episodes=2000, render_every=200)
    
    # Demo with trained agents
    input("Training complete! Press Enter to see the trained agents in action...")
    demo_trained_agents(seeker_agent, hunter_agent, episodes=5)