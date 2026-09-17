# Resting Face Judge — Baseline Report
## Phase 0 & Phase 1 Deliverable

---

## 1. Problem Definition (Phase 0)

**Input:**
A single face image (48×48 grayscale) — eventually a cropped frame from a live webcam feed.

**Output:**
A predicted facial expression class, with a confidence score.

```text
Example:
Input:  48x48 grayscale face crop
Output: happy → 68% confidence
```

**Classes (7):**
`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`

**What this model is NOT doing:**
- It does **not** identify *who* the person is (no face recognition/identity).
- It only classifies the *visible* facial expression in a single frame — it has no memory of past frames and makes no claim about the person's actual emotional state, only what their face visually resembles.

---

## 2. Dataset

- **Source:** FER-style dataset, 7 classes, pre-split train/test.
- **Train:** 28,709 images, `(48, 48)`, `uint8`, pixel range 0–255.
- **Test:** 7,178 images, same format.
- **Class imbalance confirmed:** e.g. `disgust` (111 test samples) vs. `happy` (1,774 test samples) — a ~16× imbalance between smallest and largest class.

---

## 3. Preprocessing Decisions

| Decision | Choice | Reasoning |
|---|---|---|
| Scaling | `/255.0` → normalize to [0,1] (used for Logistic Regression only) | Pixels have a known, fixed range (0–255) — no need to estimate stats like standardization requires. Simpler and more reproducible. |
| Dtype | `float32` (not default `float64`) | Halves memory vs. `float64`; ML doesn't need `float64` precision — the gain is negligible relative to real-world data noise. |
| Reshape | `.reshape(X.shape[0], -1)` → `(N, 2304)` | Dynamic shape (not hardcoded sample count) so the pipeline survives dataset-size changes without silent bugs. |
| Random Forest scaling | Raw `uint8`, no normalization | Tree splits are threshold comparisons — invariant to any monotonic rescaling. Skipping normalization is free speed/memory with no accuracy cost. |

**Known structural limitation:** flattening `(48,48)` → `(2304,)` destroys spatial adjacency between pixels. Neither model family used here can perceive that pixel *i* and pixel *i+1* might be spatially adjacent (e.g. part of the same eyebrow). This is an architectural ceiling that no amount of tuning removes — it's the core motivation for moving to a CNN in Phase 3.

---

## 4. Baseline Results

Random guessing (7 classes) ≈ **14%**. Majority-class guessing (always predict `happy`) ≈ **25%**.

| Model | Train Acc | Test Acc | Key observation |
|---|---|---|---|
| Logistic Regression (default) | — | 37.4% | `disgust` recall = 0.00 — completely missed; imbalance fully hidden by overall accuracy |
| Logistic Regression (`class_weight='balanced'`) | — | 32.5% | `disgust` recall ↑ to 0.46 (precision only 0.09 — over-predicted); `happy` recall dropped 0.68→0.42 — explicit precision/recall trade-off |
| Logistic Regression (balanced + `saga`, 3000 iter) | 46.6% | 32.7% | Took ~2h7m to train, still didn't fully converge; ~14-point train/test gap — mild overfit layered on underfit; confirmed ceiling is architectural, not an optimization problem |
| Random Forest (unlimited depth, balanced) | 99.8% | 45.8% | Severe overfitting (near-perfect train, ~46% test); best test accuracy overall; `disgust` precision 0.98 but recall only 0.38 — conservative, opposite failure mode from Logistic Regression |
| Random Forest (`max_depth=15`, balanced) | 98.6% | 44.2% | Depth constraint too weak to meaningfully reduce overfitting; test accuracy actually dipped slightly |

**Best baseline: Random Forest (unlimited depth), 45.8% test accuracy** — this is the number to beat once CNNs are introduced.

---

## 5. Conclusions

1. Both linear (Logistic Regression) and non-linear-but-spatially-blind (Random Forest) models plateau in the **33–46%** range on flattened pixel features — a ceiling driven by lost spatial structure, not by insufficient tuning.
2. Class imbalance is a persistent, real issue (`disgust` in particular) and must be checked with per-class metrics (precision/recall/F1), not accuracy alone, going forward — including once CNNs are introduced.
3. Solver/algorithm choice (`lbfgs` vs `saga`) and regularization tuning (`C`, `max_depth`) can shift *where* errors land (which class, precision vs. recall) but cannot break through the architectural ceiling.
4. Random Forest's severe overfitting (99.8% train / 45.8% test) is a useful diagnostic reference point for judging CNN training later — a much smaller gap there would indicate genuinely better generalization, not just a different failure mode.

**Baseline established. Proceeding to Phase 2 (OpenCV webcam pipeline).**
