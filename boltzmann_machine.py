"""Exemplo didático de Máquina de Boltzmann Restrita (RBM) sem dependências externas."""

from __future__ import annotations

import math
import random
from typing import List, Tuple

Vector = List[float]
Matrix = List[Vector]


class RestrictedBoltzmannMachine:
    """RBM com unidades visíveis e ocultas binárias."""

    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        learning_rate: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.rng = random.Random(seed)

        self.weights: Matrix = [
            [self.rng.uniform(-0.01, 0.01) for _ in range(n_hidden)]
            for _ in range(n_visible)
        ]
        self.visible_bias: Vector = [0.0 for _ in range(n_visible)]
        self.hidden_bias: Vector = [0.0 for _ in range(n_hidden)]

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def _sample_binary(self, prob: float) -> float:
        return 1.0 if self.rng.random() < prob else 0.0

    def sample_hidden(self, visible: Vector) -> Tuple[Vector, Vector]:
        probs: Vector = []
        states: Vector = []
        for j in range(self.n_hidden):
            activation = self.hidden_bias[j]
            for i in range(self.n_visible):
                activation += visible[i] * self.weights[i][j]
            p = self._sigmoid(activation)
            probs.append(p)
            states.append(self._sample_binary(p))
        return probs, states

    def sample_visible(self, hidden: Vector) -> Tuple[Vector, Vector]:
        probs: Vector = []
        states: Vector = []
        for i in range(self.n_visible):
            activation = self.visible_bias[i]
            for j in range(self.n_hidden):
                activation += hidden[j] * self.weights[i][j]
            p = self._sigmoid(activation)
            probs.append(p)
            states.append(self._sample_binary(p))
        return probs, states

    @staticmethod
    def _outer(vec_a: Vector, vec_b: Vector) -> Matrix:
        return [[a * b for b in vec_b] for a in vec_a]

    @staticmethod
    def _mean_squared_error(a: Vector, b: Vector) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)

    def contrastive_divergence(self, batch: Matrix) -> float:
        pos_assoc = [[0.0 for _ in range(self.n_hidden)] for _ in range(self.n_visible)]
        neg_assoc = [[0.0 for _ in range(self.n_hidden)] for _ in range(self.n_visible)]
        vis_bias_grad = [0.0 for _ in range(self.n_visible)]
        hid_bias_grad = [0.0 for _ in range(self.n_hidden)]
        recon_errors: List[float] = []

        for data in batch:
            pos_hidden_probs, pos_hidden_states = self.sample_hidden(data)
            p_outer = self._outer(data, pos_hidden_probs)

            neg_visible_probs, neg_visible_states = self.sample_visible(pos_hidden_states)
            neg_hidden_probs, _ = self.sample_hidden(neg_visible_states)
            n_outer = self._outer(neg_visible_states, neg_hidden_probs)

            for i in range(self.n_visible):
                for j in range(self.n_hidden):
                    pos_assoc[i][j] += p_outer[i][j]
                    neg_assoc[i][j] += n_outer[i][j]

            for i in range(self.n_visible):
                vis_bias_grad[i] += data[i] - neg_visible_states[i]

            for j in range(self.n_hidden):
                hid_bias_grad[j] += pos_hidden_probs[j] - neg_hidden_probs[j]

            recon_errors.append(self._mean_squared_error(data, neg_visible_probs))

        batch_size = len(batch)
        lr = self.learning_rate / batch_size

        for i in range(self.n_visible):
            for j in range(self.n_hidden):
                self.weights[i][j] += lr * (pos_assoc[i][j] - neg_assoc[i][j])

        for i in range(self.n_visible):
            self.visible_bias[i] += lr * vis_bias_grad[i]

        for j in range(self.n_hidden):
            self.hidden_bias[j] += lr * hid_bias_grad[j]

        return sum(recon_errors) / len(recon_errors)

    def train(self, data: Matrix, epochs: int = 2000, batch_size: int = 4) -> None:
        for epoch in range(1, epochs + 1):
            self.rng.shuffle(data)
            errors: List[float] = []

            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                errors.append(self.contrastive_divergence(batch))

            if epoch % 200 == 0 or epoch == 1:
                print(f"Época {epoch:4d} | erro médio: {sum(errors)/len(errors):.5f}")


def format_matrix(matrix: Matrix, digits: int = 3) -> str:
    rows = []
    for row in matrix:
        rows.append("[" + ", ".join(f"{value:.{digits}f}" for value in row) + "]")
    return "[\n  " + "\n  ".join(rows) + "\n]"


if __name__ == "__main__":
    dataset: Matrix = [
        [1, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 1, 1, 0, 1],
        [1, 1, 0, 0, 0, 0],
    ]

    rbm = RestrictedBoltzmannMachine(n_visible=6, n_hidden=2, learning_rate=0.2, seed=7)
    rbm.train(dataset.copy(), epochs=1500, batch_size=4)

    print("\nPesos aprendidos:")
    print(format_matrix(rbm.weights))

    exemplo = [1, 1, 1, 0, 0, 0]
    hidden_probs, hidden_states = rbm.sample_hidden(exemplo)
    recon_probs, _ = rbm.sample_visible(hidden_states)

    print("\nEntrada:", exemplo)
    print("Ativação oculta (prob.):", [round(v, 3) for v in hidden_probs])
    print("Reconstrução (prob.):", [round(v, 3) for v in recon_probs])
