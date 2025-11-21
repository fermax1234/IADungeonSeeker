import os
import time
import numpy as np
from IAProyecto.dungeon_seekers_env import DungeonSeekersEnv
from agents.qlearning_agent import ImprovedQLearningAgent
# training/train.py

def train_agents(
    num_maps=5,
    episodes_per_map=200,
    render_every=40,
    maze_size=15,
    maps_folder="maps"
):
    import os
    os.makedirs(maps_folder, exist_ok=True)

    seeker = ImprovedQLearningAgent(
        state_size=6 + maze_size * maze_size,
        action_size=4,
        learning_rate=0.2,
        exploration_decay=0.998
    )

    hunter = ImprovedQLearningAgent(
        state_size=6 + maze_size * maze_size,
        action_size=4,
        learning_rate=0.2,
        exploration_decay=0.998
    )

    for map_index in range(num_maps):

        print(f"\n=== TRAINING MAP {map_index} / {num_maps - 1} ===")

        # Create environment to generate valid maze
        env_gen = DungeonSeekersEnv(size=maze_size)
        maze = env_gen.maze.copy()

        # Save maze for demo
        map_path = f"{maps_folder}/map_{map_index:03d}.npy"
        np.save(map_path, maze)

        print(f"[OK] Map saved at {map_path}")

        # Training episodes for this map
        for episode in range(episodes_per_map):

            env = DungeonSeekersEnv(
                size=maze_size,
                map_index=map_index,
                total_maps=num_maps,
                loaded_maze=maze,
                episode_number=episode  # visible in pygame
            )

            state, _ = env.reset()
            done = False

            while not done:
                s_action = seeker.act(state)
                h_action = hunter.act(state)

                next_state, rewards, done, _, info = env.step(s_action, h_action)
                s_reward, h_reward = rewards

                seeker.learn(state, s_action, s_reward, next_state, done)
                hunter.learn(state, h_action, h_reward, next_state, done)

                state = next_state

                # Render training
                if episode % render_every == 0:
                    env.render()

            env.close()

        print(f"[DONE] Training completed for MAP {map_index}")

    print("\n=== TRAINING FINISHED ===")
    return seeker, hunter
