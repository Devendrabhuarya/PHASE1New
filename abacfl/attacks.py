"""Credentialed-attacker behaviours. All attackers pass the ABAC gate (valid cert)."""
import numpy as np


class Attack:
    """kind: label_flip | sign_flip | noise | scale | none
    on_off: if set to (period, on_len, start) attacker is active only when
            t >= start and (t-start) % period < on_len  (adaptive on-off attacker)."""
    def __init__(self, kind="sign_flip", scale=3.0, on_off=None):
        self.kind, self.scale, self.on_off = kind, scale, on_off

    def active(self, t):
        if self.kind == "none":
            return False
        if self.on_off is None:
            return True
        period, on_len, start = self.on_off
        return t >= start and (t - start) % period < on_len

    def poison_labels(self, y, n_cls):
        return (y + 1) % n_cls if self.kind == "label_flip" else y

    def poison_update(self, delta, rng):
        if self.kind == "sign_flip":
            return -self.scale * delta
        if self.kind == "scale":
            return self.scale * delta
        if self.kind == "noise":
            z = rng.normal(0, 1, delta.shape)
            return z / np.linalg.norm(z) * self.scale * np.linalg.norm(delta)
        if self.kind == "label_flip":
            return self.scale * delta   # optional boosting; scale=1 => pure data poisoning
        return delta
