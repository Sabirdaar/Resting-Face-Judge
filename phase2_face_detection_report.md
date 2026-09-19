# Resting Face Judge — Phase 2 Report
## Face Detection: Haar Cascade vs. OpenCV DNN

---

## 1. Goal

Before any expression classification can happen, the pipeline needs to reliably locate a face in a live webcam frame. This phase evaluates two OpenCV-native face detectors under real, deliberately-varied conditions (angle, lighting, occlusion) rather than assuming either "just works."

---

## 2. Method

Both detectors were tested live via webcam under the same informal protocol:
- Centered, frontal face at normal lighting (sanity check)
- Progressive head rotation, to find the angle where detection breaks
- Recovery behavior when returning to frontal
- Partial/full face occlusion

No formal dataset/benchmark was used — this is a live, qualitative stress test appropriate for an early pipeline decision, not a final accuracy claim.

---

## 3. Detector A: Haar Cascade (`haarcascade_frontalface_default.xml`)

**How it works:** Hand-crafted pattern matching — scans the image for a specific learned arrangement of light/dark regions (e.g. eye region darker than cheek region) characteristic of a frontal face.

**Results:**
- Reliable detection up to **~45° head rotation**.
- Beyond that angle: detection is lost.
- Recovery: **immediate** — no lag or flicker when turning back toward center.
- Failure is a **hard geometric cutoff**, not a gradual confidence drop — the detector either matches its expected frontal pattern or it doesn't.

**Interpretation:** Consistent with how Haar cascades work — they aren't reasoning about "faceness," they're matching a fixed 2D pattern. Past ~45°, that pattern (eye spacing, nose shadow position) no longer resembles what the cascade was trained to find.

**Verdict:** Not sufficient for this project. A facial-expression app cannot assume users stay near-frontal — natural reactions involve head movement.

---

## 4. Detector B: OpenCV DNN (SSD + ResNet-10, `res10_300x300_ssd_iter_140000`)

**How it works:** A real trained neural network (Single Shot Detector on a ResNet-10 backbone) outputs a bounding box **and a confidence score** per detection, rather than a binary match/no-match.

**Results:**
- Reliable detection at **significantly more difficult angles** than Haar, with high confidence scores maintained even at those angles.
- Partial occlusion (covering one eye, or the nose) — **still detected**, with a **reduced confidence score** rather than a dropped detection.
- Only real failure point: **full face occlusion**.
- Degradation is **graceful** (confidence trends down) rather than a hard cutoff.

**Verdict:** Suitable for this project. Handles the realistic range of head movement and partial occlusion expected from a live "reaction" use case.

---

## 5. Comparison Summary

| Aspect | Haar Cascade | OpenCV DNN (SSD-ResNet) |
|---|---|---|
| Detection method | Hand-crafted pattern matching | Learned neural network |
| Angle tolerance | ~45°, hard cutoff | Wide angles, high confidence maintained |
| Partial occlusion | Untested / assumed fragile (pattern-based) | Still detects, confidence drops |
| Full occlusion | Fails | Fails (expected — no visual signal left) |
| Failure style | Hard cutoff (binary) | Graceful (continuous confidence score) |
| Output | Bounding box only | Bounding box + confidence score |
| Compute cost | Very low | Low, still real-time on CPU |
| New dependencies | None (OpenCV built-in) | None (OpenCV `cv2.dnn`, model files downloaded separately) |

---

## 6. Decision

**Chosen detector: OpenCV DNN (SSD-ResNet).**

Justification: the upgrade was made *after* measuring Haar's actual failure boundary (~45°, hard cutoff) against the project's real requirement (faces at arbitrary angles during natural expression/reaction). The DNN detector's graceful confidence-based degradation and wide-angle robustness directly address that gap, at negligible added compute cost and no new external dependency.

The `CONFIDENCE_THRESHOLD` parameter (currently `0.5`) is the direct analogue of Haar's `minNeighbors`/`scaleFactor` — a precision/recall lever — and is also structurally the same kind of threshold the Phase 1 classifiers' output probabilities will need once expression prediction is added on top of this detector.

---

## 7. End-to-End Pipeline Integration


The detected face region is cropped from each frame, converted to grayscale, and resized to `48×48` to match the Phase 1 model's expected input format. The baseline Random Forest classifier model (`random_forest_fer2013.joblib`) is wired directly into the live webcam loop to prove the full end-to-end pipeline (`detect → crop → resize → predict → display`).

---

## 8. Phase 2 Limitations

> [!WARNING]
> **Documented Limitation:** Baseline RF live demo works end-to-end but exhibits expected majority-class bias toward `'happy'`, consistent with test-set behavior; domain shift between FER-2013 dataset images and live webcam frames is not yet characterized.

Key details:
- **Class Imbalance:** FER-2013 has a heavy majority representation of the `happy` class (~25% of dataset), causing the baseline Random Forest classifier to frequently default to `happy` predictions during live video inference.
- **Domain Shift:** Differences in lighting, camera resolution, sensor noise, cropping boundaries, and unconstrained real-world poses between offline FER-2013 cropped dataset images and live webcam video frames have not been formally evaluated or normalized.

---

## 9. Next Steps (Phase 3)

1. Introduce Convolutional Neural Network (CNN) architectures to preserve spatial feature hierarchies and surpass the ~45.8% baseline test accuracy ceiling.
2. Address majority-class bias and domain shift through data augmentation, class weighting, and improved feature representation.

