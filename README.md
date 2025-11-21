# 🧠 IA Dungeon Seeker — Multi-Agent Reinforcement Learning

Proyecto de aprendizaje por refuerzo multi-agente donde dos agentes (Seeker y Hunter) interactúan en laberintos generados proceduralmente, usando Q-Learning mejorado, pygame para visualización y controles interactivos en el modo demo.

---

# 📌 Requisitos del sistema

- **Python 3.10 o 3.11** (NO funciona con 3.12+)
- **Windows 10/11**
- **pygame**
- **numpy**
- **gymnasium**
- **Git (opcional si usas GitHub Desktop)**
- **GitHub Desktop (opcional)**

---

# 📦 Instalación (con entorno virtual)

## 1️⃣ Clonar el repositorio

```bash
git clone https://github.com/fermax1234/IADungeonSeeker.git
cd IADungeonSeeker
python -m venv IAProyecto
IAProyecto\Scripts\activate

Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
venv\Scripts\activate


pip install -r requirements.txt

Si no 
pip install numpy gymnasium pygame matplotlib pandas

Ejecuta
python main.py

El entrenamiento:

Genera mapas conectados automáticamente

Entrena a ambos agentes por episodios/mundos

Muestra EPISODE y STEP dentro de la ventana de pygame

Los parámetros están en main.py:
MAZE_SIZE = 15
NUM_MAPS = 3
EPISODES_PER_MAP = 200
RENDER_EVERY = 5
