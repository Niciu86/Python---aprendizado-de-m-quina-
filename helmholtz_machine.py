"""Implementação didática de uma Máquina de Helmholtz (Wake-Sleep) sem dependências externas."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random


Vector = list[float]
Matrix = list[Vector]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def vec_mat_mul(v: Vector, m: Matrix) -> Vector:
    # v: [in], m: [in][out] -> [out]
    out_dim = len(m[0])
    return [sum(v[i] * m[i][j] for i in range(len(v))) for j in range(out_dim)]


def add_vec(a: Vector, b: Vector) -> Vector:
    return [x + y for x, y in zip(a, b)]


@dataclass
class TrainHistory:
    reconstruction_error: list[float]


class HelmholtzMachine:
    def __init__(self, n_visible: int, n_hidden: int, lr_wake: float = 0.05, lr_sleep: float = 0.05, seed: int = 42) -> None:
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr_wake = lr_wake
        self.lr_sleep = lr_sleep
        self.rng = random.Random(seed)

        scale = 0.1
        self.w_rec = [[self.rng.gauss(0.0, scale) for _ in range(n_hidden)] for _ in range(n_visible)]
        self.b_rec = [0.0] * n_hidden

        self.w_gen = [[self.rng.gauss(0.0, scale) for _ in range(n_visible)] for _ in range(n_hidden)]
        self.b_gen = [0.0] * n_visible

        self.b_prior = [0.0] * n_hidden

    def sample_bernoulli(self, probs: Vector) -> Vector:
        return [1.0 if self.rng.random() < p else 0.0 for p in probs]

    def q_h_given_x(self, x: Vector) -> Vector:
        pre = add_vec(vec_mat_mul(x, self.w_rec), self.b_rec)
        return [sigmoid(z) for z in pre]

    def p_x_given_h(self, h: Vector) -> Vector:
        pre = add_vec(vec_mat_mul(h, self.w_gen), self.b_gen)
        return [sigmoid(z) for z in pre]

    def wake_phase(self, x_batch: Matrix) -> float:
        batch_size = len(x_batch)
        eps = 1e-9
        total_bce = 0.0

        dw_gen = [[0.0 for _ in range(self.n_visible)] for _ in range(self.n_hidden)]
        db_gen = [0.0] * self.n_visible
        dh_prior = [0.0] * self.n_hidden

        for x in x_batch:
            h_probs = self.q_h_given_x(x)
            h = self.sample_bernoulli(h_probs)
            x_probs = self.p_x_given_h(h)

            for j in range(self.n_visible):
                err = x[j] - x_probs[j]
                db_gen[j] += err
                for i in range(self.n_hidden):
                    dw_gen[i][j] += h[i] * err

            for i in range(self.n_hidden):
                dh_prior[i] += h[i] - sigmoid(self.b_prior[i])

            total_bce += -sum(x[j] * math.log(x_probs[j] + eps) + (1 - x[j]) * math.log(1 - x_probs[j] + eps) for j in range(self.n_visible)) / self.n_visible

        scale = self.lr_wake / batch_size
        for i in range(self.n_hidden):
            for j in range(self.n_visible):
                self.w_gen[i][j] += scale * dw_gen[i][j]
        for j in range(self.n_visible):
            self.b_gen[j] += scale * db_gen[j]
        for i in range(self.n_hidden):
            self.b_prior[i] += scale * dh_prior[i]

        return total_bce / batch_size

    def sleep_phase(self, batch_size: int) -> None:
        dw_rec = [[0.0 for _ in range(self.n_hidden)] for _ in range(self.n_visible)]
        db_rec = [0.0] * self.n_hidden

        for _ in range(batch_size):
            h_prior_probs = [sigmoid(b) for b in self.b_prior]
            h_sleep = self.sample_bernoulli(h_prior_probs)
            x_sleep_probs = self.p_x_given_h(h_sleep)
            x_sleep = self.sample_bernoulli(x_sleep_probs)

            h_pred = self.q_h_given_x(x_sleep)
            for i in range(self.n_hidden):
                err = h_sleep[i] - h_pred[i]
                db_rec[i] += err
                for j in range(self.n_visible):
                    dw_rec[j][i] += x_sleep[j] * err

        scale = self.lr_sleep / batch_size
        for j in range(self.n_visible):
            for i in range(self.n_hidden):
                self.w_rec[j][i] += scale * dw_rec[j][i]
        for i in range(self.n_hidden):
            self.b_rec[i] += scale * db_rec[i]

    def train(self, data: Matrix, epochs: int = 120, batch_size: int = 64) -> TrainHistory:
        history = TrainHistory(reconstruction_error=[])

        for epoch in range(1, epochs + 1):
            shuffled = data[:]
            self.rng.shuffle(shuffled)
            losses: list[float] = []

            for start in range(0, len(shuffled), batch_size):
                batch = shuffled[start : start + batch_size]
                if not batch:
                    continue
                loss = self.wake_phase(batch)
                self.sleep_phase(len(batch))
                losses.append(loss)

            mean_loss = sum(losses) / len(losses)
            history.reconstruction_error.append(mean_loss)
            if epoch % 20 == 0 or epoch == 1:
                print(f"Época {epoch:03d} | erro de reconstrução (BCE): {mean_loss:.4f}")

        return history

    def reconstruct(self, x: Vector) -> Vector:
        h_probs = self.q_h_given_x(x)
        h = [1.0 if p > 0.5 else 0.0 for p in h_probs]
        return self.p_x_given_h(h)


def make_synthetic_binary_patterns(samples: int = 1200, side: int = 8, seed: int = 7) -> Matrix:
    rng = random.Random(seed)
    data: Matrix = []

    for _ in range(samples):
        image = [[0.0 for _ in range(side)] for _ in range(side)]
        if rng.random() < 0.5:
            row = rng.randrange(side)
            for c in range(side):
                image[row][c] = 1.0
        else:
            col = rng.randrange(side)
            for r in range(side):
                image[r][col] = 1.0

        for r in range(side):
            for c in range(side):
                if rng.random() < 0.05:
                    image[r][c] = 1.0 - image[r][c]

        flat = [pix for row in image for pix in row]
        data.append(flat)

    return data


if __name__ == "__main__":
    X = make_synthetic_binary_patterns(samples=400, side=8)

    model = HelmholtzMachine(n_visible=64, n_hidden=12, lr_wake=0.08, lr_sleep=0.05, seed=123)
    model.train(X, epochs=10, batch_size=32)

    print("\nReconstrução (5 amostras, 10 primeiras dimensões):")
    for sample in X[:5]:
        recon = model.reconstruct(sample)
        print([round(v, 3) for v in recon[:10]])
