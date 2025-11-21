from training.train import train_agents
from training.demo import demo_trained_agents

# ====== METADATOS ======
MAZE_SIZE = 15           # Tamaño del laberinto
NUM_MAPS = 3             # Cantidad de mapas a entrenar
EPISODES_PER_MAP = 200   # Episodios por mapa
RENDER_EVERY = 5        # Cada cuántos episodios mostrar animación




if __name__ == "__main__":

    seeker, hunter = train_agents(
        num_maps=NUM_MAPS,
        episodes_per_map=EPISODES_PER_MAP,
        render_every=RENDER_EVERY,
        maze_size=MAZE_SIZE
    )

    input("\nTraining completed. Press ENTER to start demo...")

    all_maps = list(range(NUM_MAPS))

    demo_trained_agents(
        seeker_agent=seeker,
        hunter_agent=hunter,
        map_indices=all_maps,
        maze_size=MAZE_SIZE
    )