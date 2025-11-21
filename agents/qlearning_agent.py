# agents/qlearning_agent.py

import numpy as np
from collections import defaultdict


class ImprovedQLearningAgent:
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=1.0,
        exploration_decay=0.997,
        exploration_min=0.01
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.learning_rate_decay = 0.9995
        self.min_learning_rate = 0.01

        # Use partial state for Q-table keys
        self.q_table = defaultdict(lambda: np.zeros(action_size))

    def act(self, state):
        key = tuple(state[:10])
        if np.random.random() < self.exploration_rate:
            return np.random.randint(self.action_size)
        return np.argmax(self.q_table[key])

    def learn(self, state, action, reward, next_state, done):
        key = tuple(state[:10])
        next_key = tuple(next_state[:10])

        current_q = self.q_table[key][action]

        target = reward if done else reward + self.discount_factor * np.max(self.q_table[next_key])

        self.q_table[key][action] += self.learning_rate * (target - current_q)

        if done:
            self.exploration_rate = max(
                self.exploration_min,
                self.exploration_rate * self.exploration_decay
            )
            self.learning_rate = max(
                self.min_learning_rate,
                self.learning_rate * self.learning_rate_decay
            )
