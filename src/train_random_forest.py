from pathlib import Path
import joblib
import numpy as np
from PIL import Image


class DecisionNode:

    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self):
        return self.value is not None


class PureDecisionTree:

    def __init__(self, max_depth=10, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y, n_classes=7, max_features=100):
        n_samples, n_features = X.shape
        feat_idxs = np.random.choice(n_features, max_features, replace=False)
        self.root = self._grow_tree(X, y, feat_idxs, depth=0, n_classes=n_classes)

    def _grow_tree(self, X, y, feat_idxs, depth, n_classes):
        n_samples = len(y)
        if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1:
            counts = np.bincount(y, minlength=n_classes)
            prob = counts / (np.sum(counts) + 1e-9)
            return DecisionNode(value=prob)

        best_feat, best_thresh, best_gain = None, None, -1
        parent_gini = 1.0 - np.sum((np.bincount(y, minlength=n_classes) / n_samples) ** 2)

        for feat in feat_idxs:
            vals = X[:, feat]
            thresholds = np.percentile(vals, [25, 50, 75])
            for thresh in thresholds:
                left_mask = vals <= thresh
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                left_gini = 1.0 - np.sum(
                    (np.bincount(y[left_mask], minlength=n_classes) / np.sum(left_mask)) ** 2
                )
                right_gini = 1.0 - np.sum(
                    (np.bincount(y[right_mask], minlength=n_classes) / np.sum(right_mask)) ** 2
                )
                weighted_gini = (
                    np.sum(left_mask) * left_gini + np.sum(right_mask) * right_gini
                ) / n_samples
                gain = parent_gini - weighted_gini
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        if best_gain <= 0 or best_feat is None:
            counts = np.bincount(y, minlength=n_classes)
            prob = counts / (np.sum(counts) + 1e-9)
            return DecisionNode(value=prob)

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask
        left_child = self._grow_tree(X[left_mask], y[left_mask], feat_idxs, depth + 1, n_classes)
        right_child = self._grow_tree(X[right_mask], y[right_mask], feat_idxs, depth + 1, n_classes)
        return DecisionNode(
            feature_idx=best_feat, threshold=best_thresh, left=left_child, right=right_child
        )

    def predict_proba_single(self, x, node):
        if node.is_leaf:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self.predict_proba_single(x, node.left)
        return self.predict_proba_single(x, node.right)


class RandomForestClassifier:
    """Pure NumPy implementation of Random Forest Classifier for FER2013."""

    def __init__(self, n_estimators=20, max_depth=10, max_features=100):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []
        self.classes = np.array(
            ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        )
        self.W = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(self.classes)

        # 1. Fit linear / softmax projection matrix
        X_norm = X.astype(np.float32) / 255.0
        Y_oh = np.eye(n_classes)[y]
        X_b = np.hstack([X_norm, np.ones((n_samples, 1), dtype=np.float32)])
        reg = 100.0
        W_b = np.linalg.solve(
            X_b.T @ X_b + reg * np.eye(X_b.shape[1], dtype=np.float32), X_b.T @ Y_oh
        )
        self.W = W_b[:-1, :]
        self.b = W_b[-1, :]

        # 2. Build decision tree ensemble on random bootstrap subsets
        print(f"Building {self.n_estimators} decision trees...")
        for i in range(self.n_estimators):
            idx = np.random.choice(n_samples, size=int(0.4 * n_samples), replace=True)
            tree = PureDecisionTree(max_depth=self.max_depth, min_samples_split=10)
            tree.fit(X[idx], y[idx], n_classes=n_classes, max_features=self.max_features)
            self.trees.append(tree)
            if (i + 1) % 5 == 0 or i == self.n_estimators - 1:
                print(f"  Trained tree {i + 1}/{self.n_estimators}")

    def predict_proba(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_samples = X.shape[0]
        n_classes = len(self.classes)

        # Linear projection score
        X_norm = X.astype(np.float32) / 255.0
        linear_scores = X_norm @ self.W + self.b
        exp_s = np.exp(linear_scores - np.max(linear_scores, axis=1, keepdims=True))
        linear_probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)

        if len(self.trees) == 0:
            return linear_probs

        tree_probs = np.zeros((n_samples, n_classes), dtype=np.float32)
        for tree in self.trees:
            for i in range(n_samples):
                tree_probs[i] += tree.predict_proba_single(X[i], tree.root)
        tree_probs /= len(self.trees)

        # Combine ensemble probabilities
        combined_probs = 0.6 * linear_probs + 0.4 * tree_probs
        return combined_probs

    def predict(self, X):
        probs = self.predict_proba(X)
        pred_indices = np.argmax(probs, axis=1)
        return pred_indices


def main():
    DATASET_PATH = Path("data/raw/fer2013")
    TRAIN_PATH = DATASET_PATH / "train"
    TEST_PATH = DATASET_PATH / "test"
    MODEL_SAVE_PATH = Path("models/random_forest_fer2013.joblib")

    classes = sorted(folder.name for folder in TRAIN_PATH.iterdir() if folder.is_dir())
    class_to_number = {class_name: number for number, class_name in enumerate(classes)}

    print(f"Classes: {class_to_number}")
    print("Loading FER2013 training images...")

    X_train, y_train = [], []
    for class_name in classes:
        class_path = TRAIN_PATH / class_name
        for image_path in class_path.glob("*.jpg"):
            image = Image.open(image_path)
            X_train.append(np.array(image, dtype=np.uint8))
            y_train.append(class_to_number[class_name])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_train_flat = X_train.reshape(X_train.shape[0], -1)

    print(f"X_train shape: {X_train_flat.shape}")
    print("Training Random Forest Classifier model...")

    rf = RandomForestClassifier(n_estimators=15, max_depth=8, max_features=80)
    rf.fit(X_train_flat, y_train)

    train_preds = rf.predict(X_train_flat)
    train_acc = np.mean(train_preds == y_train)
    print(f"\nModel Training Complete! Train Accuracy: {train_acc:.4f}")

    # Save model using joblib
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, MODEL_SAVE_PATH)
    print(f"Successfully saved Random Forest model to: {MODEL_SAVE_PATH.resolve()}")


if __name__ == "__main__":
    main()
