"""A small, correct, from-scratch neural-network library in NumPy.

CROSSING's temporal models (crossing-intention, learned trajectory refinement) are genuine
sequence networks — a **GRU with full back-propagation-through-time** and an Adam optimiser
— implemented here without any deep-learning framework, so the whole platform trains and
runs on the confirmed NumPy/scikit-learn stack. If ``torch`` is available it can be swapped
in, but nothing depends on it.

The GRU math is standard (Cho et al., 2014); the backward pass is verified against finite
differences in the test suite (``tests/test_nn_gradcheck.py``).
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------------------
# activations
# --------------------------------------------------------------------------------------
def sigmoid(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def dtanh(y):          # derivative given y = tanh(x)
    return 1.0 - y * y


def bce_loss(prob, y, eps=1e-7):
    prob = np.clip(prob, eps, 1 - eps)
    return float(-np.mean(y * np.log(prob) + (1 - y) * np.log(1 - prob)))


# --------------------------------------------------------------------------------------
# Adam
# --------------------------------------------------------------------------------------
class Adam:
    def __init__(self, params: dict, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8, wd=1e-5):
        self.p = params
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: dict):
        self.t += 1
        for k in self.p:
            g = grads[k] + self.wd * self.p[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# --------------------------------------------------------------------------------------
# GRU sequence classifier / regressor
# --------------------------------------------------------------------------------------
class GRU:
    """Batched GRU over fixed-length sequences with an affine readout from the last state.

    Input  X : (B, T, D)      (masked/padded to length T)
    Output   : (B, out) logits (classification) or values (regression)
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        s_in = 1.0 / np.sqrt(in_dim)
        s_h = 1.0 / np.sqrt(hidden)
        self.H = hidden
        self.p = {
            "Wz": rng.uniform(-s_in, s_in, (in_dim, hidden)),
            "Wr": rng.uniform(-s_in, s_in, (in_dim, hidden)),
            "Wh": rng.uniform(-s_in, s_in, (in_dim, hidden)),
            "Uz": rng.uniform(-s_h, s_h, (hidden, hidden)),
            "Ur": rng.uniform(-s_h, s_h, (hidden, hidden)),
            "Uh": rng.uniform(-s_h, s_h, (hidden, hidden)),
            "bz": np.zeros(hidden), "br": np.zeros(hidden), "bh": np.zeros(hidden),
            "Wo": rng.uniform(-s_h, s_h, (hidden, out_dim)),
            "bo": np.zeros(out_dim),
        }

    # -- forward ----------------------------------------------------------------------
    def forward(self, X):
        B, T, D = X.shape
        p = self.p
        h = np.zeros((B, self.H))
        cache = []
        for t in range(T):
            x = X[:, t, :]
            z = sigmoid(x @ p["Wz"] + h @ p["Uz"] + p["bz"])
            r = sigmoid(x @ p["Wr"] + h @ p["Ur"] + p["br"])
            hh = np.tanh(x @ p["Wh"] + (r * h) @ p["Uh"] + p["bh"])
            h_new = (1 - z) * h + z * hh
            cache.append((x, h, z, r, hh, h_new))
            h = h_new
        logits = h @ p["Wo"] + p["bo"]
        self._cache = (X, cache, h)
        return logits

    # -- backward (BPTT) --------------------------------------------------------------
    def backward(self, dlogits):
        X, cache, hT = self._cache
        p = self.p
        B, T, D = X.shape
        g = {k: np.zeros_like(v) for k, v in p.items()}
        g["Wo"] += hT.T @ dlogits
        g["bo"] += dlogits.sum(axis=0)
        dh = dlogits @ p["Wo"].T
        for t in reversed(range(T)):
            x, h_prev, z, r, hh, h_new = cache[t]
            dz = dh * (hh - h_prev)
            dhh = dh * z
            dh_prev = dh * (1 - z)
            # through hh = tanh(a_h), a_h = x Wh + (r*h_prev) Uh + bh
            da_h = dhh * dtanh(hh)
            g["Wh"] += x.T @ da_h
            g["bh"] += da_h.sum(axis=0)
            drh = da_h @ p["Uh"].T           # d(r*h_prev)
            g["Uh"] += (r * h_prev).T @ da_h
            dr = drh * h_prev
            dh_prev += drh * r
            # through z = sigmoid(a_z)
            da_z = dz * z * (1 - z)
            g["Wz"] += x.T @ da_z
            g["Uz"] += h_prev.T @ da_z
            g["bz"] += da_z.sum(axis=0)
            dh_prev += da_z @ p["Uz"].T
            # through r = sigmoid(a_r)
            da_r = dr * r * (1 - r)
            g["Wr"] += x.T @ da_r
            g["Ur"] += h_prev.T @ da_r
            g["br"] += da_r.sum(axis=0)
            dh_prev += da_r @ p["Ur"].T
            dh = dh_prev
        return g

    # -- convenience: binary classification -------------------------------------------
    def predict_proba(self, X):
        return sigmoid(self.forward(X))[:, 0]

    def save(self, path):
        np.savez(path, **{k: v for k, v in self.p.items()}, _H=np.array([self.H]))

    @staticmethod
    def load(path):
        d = np.load(path)
        H = int(d["_H"][0])
        in_dim = d["Wz"].shape[0]
        out_dim = d["Wo"].shape[1]
        g = GRU(in_dim, H, out_dim)
        for k in g.p:
            g.p[k] = d[k]
        return g


def train_gru_classifier(gru: GRU, X, y, epochs=40, lr=3e-3, batch=64, seed=0,
                         val=None, verbose=False, class_weight=None):
    """Train a GRU for binary classification with BCE + Adam. Returns history dict."""
    rng = np.random.default_rng(seed)
    opt = Adam(gru.p, lr=lr)
    y = y.astype(float)
    n = len(X)
    hist = {"loss": [], "val_auc": []}
    for ep in range(epochs):
        idx = rng.permutation(n)
        losses = []
        for i in range(0, n, batch):
            bi = idx[i:i + batch]
            xb, yb = X[bi], y[bi]
            logits = gru.forward(xb)
            prob = sigmoid(logits[:, 0])
            # weighted BCE gradient wrt logits: d/dlogit = (prob - y) * w / N
            w = np.ones_like(yb)
            if class_weight is not None:
                w = np.where(yb > 0.5, class_weight[1], class_weight[0])
            dlogit = ((prob - yb) * w / len(bi))[:, None]
            grads = gru.backward(dlogit)
            opt.step(grads)
            losses.append(bce_loss(prob, yb))
        hist["loss"].append(float(np.mean(losses)))
        if val is not None:
            from sklearn.metrics import roc_auc_score
            pv = _predict_proba_batched(gru, val[0])
            try:
                hist["val_auc"].append(float(roc_auc_score(val[1], pv)))
            except ValueError:
                hist["val_auc"].append(float("nan"))
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            extra = f" val_auc={hist['val_auc'][-1]:.3f}" if val is not None else ""
            print(f"  epoch {ep:3d} loss={hist['loss'][-1]:.4f}{extra}")
    return hist


def _predict_proba_batched(gru: GRU, X, bs=256):
    out = []
    for i in range(0, len(X), bs):
        out.append(gru.predict_proba(X[i:i + bs]))
    return np.concatenate(out) if out else np.zeros(0)
