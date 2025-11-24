import numpy as np
from collections import deque


def bfs_path_exists(maze, start, goal):
    """Return True if a path exists from start to goal in the maze."""
    h, w = maze.shape
    q = deque([start])
    visited = {tuple(start)}

    while q:
        x, y = q.popleft()

        if (x, y) == tuple(goal):
            return True

        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                if maze[ny, nx] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append((nx, ny))

    return False


def maze_score(maze):
    """
    Score the maze. Higher is better.
    Evaluates:
    - open cell percentage
    - reasonable obstacle density
    """
    h, w = maze.shape
    open_cells = np.sum(maze == 0)
    open_ratio = open_cells / (h * w)

    # Too many or too few walls → bad maze
    if open_ratio < 0.50 or open_ratio > 0.90:
        return -1

    return open_ratio * 100


def validate_maze(maze, start, key, exit_pos, min_score=15):
    """Check if maze is valid using BFS and scoring."""
    if not bfs_path_exists(maze, start, key):
        return False

    if not bfs_path_exists(maze, key, exit_pos):
        return False

    if maze_score(maze) < min_score:
        return False

    return True
