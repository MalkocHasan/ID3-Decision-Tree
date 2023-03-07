import math
from collections import Counter

class DecisionTree:
    
    def __init__(self, max_depth=None, min_samples_split=2, min_gain=0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        
    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)
        
    def predict(self, X):
        return [self._predict(inputs) for inputs in X]
        
    def _best_split(self, X, y):
        m = y.size
        if m <= 1:
            return None, None
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
        best_gain = -1
        best_feature = None
        best_threshold = None
        for f in range(self.n_features_):
            values = X[:, f]
            unique_values = set(values)
            for threshold in unique_values:
                g_left = [np.sum(y[values <= threshold] == c) for c in range(self.n_classes_)]
                g_right = [np.sum(y[values > threshold] == c) for c in range(self.n_classes_)]
                n_left = np.sum(values <= threshold)
                n_right = m - n_left
                gain = self._information_gain(num_parent, [g_left, g_right], [n_left, n_right])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = f
                    best_threshold = threshold
        if best_gain > self.min_gain:
            return best_feature, best_threshold
        else:
            return None, None
        
    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(predicted_class=predicted_class)
        if depth < self.max_depth:
            feature, threshold = self._best_split(X, y)
            if feature is not None:
                indices_left = X[:, feature] <= threshold
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                if len(X_left) > self.min_samples_split and len(X_right) > self.min_samples_split:
                    node.feature = feature
                    node.threshold = threshold
                    node.left = self._grow_tree(X_left, y_left, depth+1)
                    node.right = self._grow_tree(X_right, y_right, depth+1)
        return node
        
    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class
        
    def _entropy(self, y):
        m = y.size
        if m == 0:
            return 0
        p = np.array([np.sum(y == c) / m for c in range(self.n_classes_)])
        return -np.sum(p * np.log2(p))
        
    def _information_gain(self, parent, children, n_samples):
        if sum(n_samples) == 0:
            return 0
        weighted_avg_entropy = sum(n / sum(n_samples) * self._entropy(y) for n, y in zip(children,
