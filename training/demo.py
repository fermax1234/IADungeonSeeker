import numpy as np
import pygame
import time
from IAProyecto.dungeon_seekers_env import DungeonSeekersEnv

# training/demo.py

def show_menu(screen, font, map_index):
    """Muestra el menú después de cada mapa."""
    screen.fill((0, 0, 0))

    options = [
        f"Mapa {map_index} finalizado.",
        "",
        "[R] Repetir este mapa",
        "[N] Siguiente mapa",
        "[Q] Salir del demo"
    ]

    y = 100
    for line in options:
        text_surf = font.render(line, True, (255, 255, 255))
        screen.blit(text_surf, (50, y))
        y += 50

    pygame.display.flip()

    # Esperar input
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    return "repeat"
                if event.key == pygame.K_n:
                    return "next"
                if event.key == pygame.K_q:
                    return "quit"


def demo_trained_agents(
    seeker_agent,
    hunter_agent,
    map_indices,
    maze_size=15,
    maps_folder="maps"
):
    print("\n=== DEMO DE AGENTES ENTRENADOS ===")

    pygame.font.init()
    menu_font = pygame.font.SysFont("Arial", 36)

    current_map_idx = 0

    while current_map_idx < len(map_indices):
        map_index = map_indices[current_map_idx]
        print(f"\n--- DEMO MAPA {map_index} ---")

        maze_path = f"{maps_folder}/map_{map_index:03d}.npy"
        maze = np.load(maze_path)

        # Ejecutar DEMO de este mapa
        repeat_map = True

        while repeat_map:
            env = DungeonSeekersEnv(
                size=maze_size,
                loaded_maze=maze,
                map_index=map_index,
                episode_number=0
            )

            state, _ = env.reset()
            done = False
            step = 0

            while not done:
                key = tuple(state[:10])
                s_action = np.argmax(seeker_agent.q_table[key])
                h_action = np.argmax(hunter_agent.q_table[key])

                next_state, _, done, _, info = env.step(s_action, h_action)
                state = next_state

                step += 1
                env.steps = step
                env.render()

                # Procesar eventos básicos (cerrar ventana)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

            # DEMO terminado → MENÚ EN PANTALLA
            decision = show_menu(env.screen, menu_font, map_index)
            env.close()

            if decision == "repeat":
                continue
            elif decision == "next":
                repeat_map = False
            elif decision == "quit":
                pygame.quit()
                return

        current_map_idx += 1

    print("\n=== DEMO FINALIZADO ===")
    pygame.quit()
