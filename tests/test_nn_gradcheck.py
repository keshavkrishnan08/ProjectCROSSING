"""Gradient check the from‑scratch GRU's BPTT against finite differences."""
import numpy as np

from crossing.ml.nn import GRU, sigmoid, bce_loss, train_gru_classifier


def test_gru_bptt_matches_finite_difference():
    rng = np.random.default_rng(0)
    B, T, D, H = 4, 5, 3, 6
    X = rng.normal(size=(B, T, D))
    y = rng.integers(0, 2, size=B).astype(float)
    g = GRU(D, H, 1, seed=1)

    logits = g.forward(X)
    prob = sigmoid(logits[:, 0])
    grads = g.backward(((prob - y) / B)[:, None])

    eps, worst = 1e-5, 0.0
    for name in ["Wz", "Uh", "Wo", "bh", "bo", "Ur", "Wr", "Uz"]:
        P, G = g.p[name], grads[name]
        for idx in list(np.ndindex(P.shape))[:5]:
            old = P[idx]
            P[idx] = old + eps; Lp = bce_loss(sigmoid(g.forward(X)[:, 0]), y)
            P[idx] = old - eps; Lm = bce_loss(sigmoid(g.forward(X)[:, 0]), y)
            P[idx] = old
            num = (Lp - Lm) / (2 * eps)
            rel = abs(num - G[idx]) / (abs(num) + abs(G[idx]) + 1e-9)
            worst = max(worst, rel)
    assert worst < 1e-4, f"max relative grad error {worst}"


def test_gru_learns_separable_sequences():
    # class 1 sequences trend up in feature 0; class 0 trend down. GRU should separate.
    rng = np.random.default_rng(3)
    T, D = 8, 2
    X, y = [], []
    for i in range(200):
        c = i % 2
        base = np.linspace(0, 1 if c else -1, T)[:, None]
        seq = np.concatenate([base, rng.normal(0, 0.1, (T, 1))], axis=1)
        X.append(seq + rng.normal(0, 0.05, (T, D))); y.append(c)
    X = np.array(X); y = np.array(y, float)
    g = GRU(D, 8, 1, seed=0)
    train_gru_classifier(g, X, y, epochs=30, lr=5e-3, batch=32)
    acc = ((g.predict_proba(X) > 0.5).astype(int) == y).mean()
    assert acc > 0.9
