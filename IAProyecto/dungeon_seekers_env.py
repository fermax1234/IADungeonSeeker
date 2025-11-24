import gymnasium as gym
import numpy as np
import pygame
from maze_generator import generate_valid_maze


class DungeonSeekersEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, size=15, map_index=0, total_maps=1,
                 loaded_maze=None, episode_number=0):
        super().__init__()

        self.size = size
        self.map_index = map_index
        self.total_maps = total_maps
        self.loaded_maze = loaded_maze
        self.episode_number = episode_number

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0, high=size - 1,
            shape=(6 + size * size,),
            dtype=np.int32
        )

        self._init_pygame()
        self.reset()

    def _init_pygame(self):
        pygame.init()
        self.cell_size = 40
        self.screen = pygame.display.set_mode(
            (self.size * self.cell_size, self.size * self.cell_size)
        )
        pygame.display.set_caption("Dungeon Seekers RL")
        self.font = pygame.font.SysFont("Arial", 28)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.loaded_maze is not None:
            self.maze = self.loaded_maze.copy()
        else:
            self.maze = generate_valid_maze(self.size)

        self.seeker_pos = np.array([1, 1])
        self.hunter_pos = np.array([self.size - 2, self.size - 2])
        self.key_pos = np.array([self.size // 2, self.size // 2])

        self.seeker_has_key = False
        self.steps = 0
        self.max_steps = 300

        return self._get_observation(), {}

    def _get_observation(self):
        base = np.array([
            self.seeker_pos[0], self.seeker_pos[1],
            self.hunter_pos[0], self.hunter_pos[1],
            self.key_pos[0], self.key_pos[1],
            int(self.seeker_has_key)
        ])
        flat = self.maze.flatten()
        return np.concatenate([base, flat])

    def step(self, seeker_action, hunter_action):
        self.steps += 1
        reward_s, reward_h = -0.05, -0.05
        done = False
        info = {}

        new_s = self._move(self.seeker_pos, seeker_action)
        if self._valid(new_s):
            self.seeker_pos = new_s

        new_h = self._move(self.hunter_pos, hunter_action)
        if self._valid(new_h):
            self.hunter_pos = new_h

        if not self.seeker_has_key and np.array_equal(self.seeker_pos, self.key_pos):
            self.seeker_has_key = True
            reward_s += 10
            info["key_found"] = True

        if self.seeker_has_key and np.array_equal(self.seeker_pos, [1, 1]):
            reward_s += 30
            done = True
            info["mission_complete"] = True

        if np.array_equal(self.seeker_pos, self.hunter_pos):
            reward_h += 20
            reward_s -= 10
            done = True
            info["seeker_caught"] = True

        if self.steps >= self.max_steps:
            done = True
            info["timeout"] = True

        return self._get_observation(), (reward_s, reward_h), done, False, info

    def _move(self, pos, action):
        x, y = pos
        if action == 0: y -= 1
        elif action == 1: x += 1
        elif action == 2: y += 1
        elif action == 3: x -= 1
        return np.array([x, y])

    def _valid(self, pos):
        x, y = pos
        return (
            0 <= x < self.size and
            0 <= y < self.size and
            self.maze[y, x] == 0
        )

    def render(self):
        self.screen.fill((50, 50, 50))

        for y in range(self.size):
            for x in range(self.size):
                if self.maze[y, x] == 1:
                    pygame.draw.rect(
                        self.screen, (255, 255, 255),
                        (x * self.cell_size, y * self.cell_size,
                         self.cell_size, self.cell_size)
                    )

        pygame.draw.rect(
            self.screen, (255, 215, 0),
            (self.key_pos[0] * self.cell_size,
             self.key_pos[1] * self.cell_size,
             self.cell_size, self.cell_size)
        )

        pygame.draw.circle(
            self.screen, (0, 0, 255),
            ((self.seeker_pos[0] + 0.5) * self.cell_size,
             (self.seeker_pos[1] + 0.5) * self.cell_size),
            self.cell_size // 3
        )

        pygame.draw.circle(
            self.screen, (255, 0, 0),
            ((self.hunter_pos[0] + 0.5) * self.cell_size,
             (self.hunter_pos[1] + 0.5) * self.cell_size),
            self.cell_size // 3
        )

        # === HUD TEXT OVERLAY ===
        hud_bg = (60, 60, 60)
        pygame.draw.rect(
            self.screen,
            hud_bg,
            pygame.Rect(0, 0, self.size * self.cell_size, 40)
        )

        episode_text = f"EPISODE: {self.episode_number}"
        step_text = f"STEP: {self.steps}"

        episode_surface = self.font.render(episode_text, True, (255, 255, 255))
        step_surface = self.font.render(step_text, True, (255, 255, 255))

        self.screen.blit(episode_surface, (10, 8))
        self.screen.blit(step_surface, (250, 8))


        pygame.display.flip()
