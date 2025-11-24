import numpy as np
import random
from utils_pathfinding import validate_maze


def generate_candidate_maze(size, obstacle_ratio=0.25):
    """Generates random maze with perimeter walls and obstacles."""
    maze = np.zeros((size, size), dtype=int)
    
    maze[0, :] = maze[-1, :] = 1
    maze[:, 0] = maze[:, -1] = 1

    obstacles = int(size * size * obstacle_ratio)

    for _ in range(obstacles):
        x = random.randint(1, size - 2)
        y = random.randint(1, size - 2)
        maze[y, x] = 1

    return maze


def generate_valid_maze(size, max_attempts=200):
    """
    Try generating mazes until one is valid based on BFS + scoring.
    """
    start = (1, 1)
    exit_pos = (size - 2, size - 2)
    key = (size // 2, size // 2)

    for attempt in range(max_attempts):
        maze = generate_candidate_maze(size)

        if validate_maze(maze, start, key, exit_pos):
            print(f"[OK] Valid maze generated in attempt {attempt+1}")
            return maze

    raise RuntimeError("Cannot generate valid maze.")
