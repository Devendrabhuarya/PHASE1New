"""Tiny numpy MLP + local training. Flat parameter vector so updates are plain vectors."""
import numpy as np


class MLP:
    def __init__(self, d_in, d_h, n_cls):
        self.d_in, self.d_h, self.n_cls = d_in, d_h, n_cls
        self.sizes = [d_in * d_h, d_h, d_h * n_cls, n_cls]
        self.n_params = sum(self.sizes)

    def init(self, rng):
        W1 = rng.normal(0, np.sqrt(2 / self.d_in), (self.d_in, self.d_h))
        W2 = rng.normal(0, np.sqrt(2 / self.d_h), (self.d_h, self.n_cls))
        return np.concatenate([W1.ravel(), np.zeros(self.d_h), W2.ravel(), np.zeros(self.n_cls)])

    def unpack(self, w):
        a, b, c, _ = np.cumsum(self.sizes)[:4]
        W1 = w[:a].reshape(self.d_in, self.d_h)
        b1 = w[a:b]
        W2 = w[b:c].reshape(self.d_h, self.n_cls)
        b2 = w[c:]
        return W1, b1, W2, b2

    def logits(self, w, X):
        W1, b1, W2, b2 = self.unpack(w)
        return np.maximum(X @ W1 + b1, 0) @ W2 + b2

    def grad(self, w, X, y):
        W1, b1, W2, b2 = self.unpack(w)
        z1 = X @ W1 + b1
        h = np.maximum(z1, 0)
        lg = h @ W2 + b2
        lg -= lg.max(1, keepdims=True)
        p = np.exp(lg)
        p /= p.sum(1, keepdims=True)
        n = len(X)
        p[np.arange(n), y] -= 1
        p /= n
        gW2, gb2 = h.T @ p, p.sum(0)
        dh = p @ W2.T
        dh[z1 <= 0] = 0
        gW1, gb1 = X.T @ dh, dh.sum(0)
        return np.concatenate([gW1.ravel(), gb1, gW2.ravel(), gb2])

    def local_train(self, w, X, y, epochs, lr, bs, rng):
        w = w.copy()
        n = len(X)
        for _ in range(epochs):
            idx = rng.permutation(n)
            for s in range(0, n, bs):
                b = idx[s:s + bs]
                w -= lr * self.grad(w, X[b], y[b])
        return w

    def predict(self, w, X):
        return self.logits(w, X).argmax(1)


def macro_f1(y_true, y_pred, n_cls):
    f1s = []
    for c in range(n_cls):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        d = 2 * tp + fp + fn
        f1s.append(2 * tp / d if d > 0 else 0.0)
    return float(np.mean(f1s))
