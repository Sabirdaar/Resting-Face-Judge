# Resting Face Judge — Phase 3 Progress Notes
## CNN Data Pipeline (in progress)

---

## Framework
**TensorFlow/Keras** — chosen over PyTorch. High-level `Sequential`/`model.fit()` API, single portable save format (`.keras`), integrates cleanly with the FastAPI deployment planned for Phase 4.

---

## Input Shape

CNNs (`Conv2D` layers) require an explicit channel dimension, even for grayscale — Keras expects `(N, H, W, C)`, not `(N, H, W)`.

```python
X_train_cnn = X_train.reshape(*X_train.shape, 1)   # (28709, 48, 48) -> (28709, 48, 48, 1)
X_test_cnn = X_test.reshape(*X_test.shape, 1)
```

Unlike Phase 1's `.reshape(N, -1)`, this **preserves** spatial structure rather than destroying it — the CNN is specifically built to exploit the 2D adjacency that Logistic Regression/Random Forest couldn't see.

---

## Normalization

CNNs train via gradient descent (backprop) — same reasoning as Phase 1's Logistic Regression applies directly: unscaled 0–255 inputs distort the loss surface and slow convergence.

```python
X_train_cnn = X_train_cnn.astype('float32') / 255.0
X_test_cnn = X_test_cnn.astype('float32') / 255.0
```

**Key rule carried forward from Phase 1:** whatever transform is applied to training data must be applied identically to test (and later, live webcam) data. Trivial here since `/255.0` is a fixed, data-independent transform — no fitted statistics involved, so no data-leakage risk. This becomes a real risk only if a data-dependent scaler (e.g. `StandardScaler`) is introduced later — noted as a thing to watch for, not something in use now.

---

## Label Encoding

Chose **one-hot encoding** (`categorical_crossentropy`) over integer labels (`sparse_categorical_crossentropy`) — more explicit, matches most tutorial/reference code for easier cross-referencing.

```python
from tensorflow.keras.utils import to_categorical

y_train_onehot = to_categorical(y_train, num_classes=7)
y_test_onehot = to_categorical(y_test, num_classes=7)
```

Sanity check: `y_train_onehot.shape` should be `(28709, 7)` — mismatch here would flag a `num_classes` error (e.g. if a class were ever dropped from the dataset).

---

## Full Pipeline So Far

```python
X_train_cnn = X_train.reshape(*X_train.shape, 1).astype('float32') / 255.0
X_test_cnn = X_test.reshape(*X_test.shape, 1).astype('float32') / 255.0

y_train_onehot = to_categorical(y_train, num_classes=7)
y_test_onehot = to_categorical(y_test, num_classes=7)
```

**Result:** `(N, 48, 48, 1)` in, `(N, 7)` out — both scaled/encoded, no leakage risk.

---

## Next (not yet decided)

- Loss function / optimizer choice for `model.compile(...)`
- Architecture shape (`Conv2D`/`Dense` layer count) — to be kept high-level/MLOps-focused per project scope, not a deep architecture deep-dive
