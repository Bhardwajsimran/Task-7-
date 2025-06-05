import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load dataset
cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce to 2D using PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Train SVM with linear kernel
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train, y_train)

# Train SVM with RBF kernel
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_train, y_train)

# Evaluate both models
print("Linear SVM Accuracy:", svm_linear.score(X_test, y_test))
print("RBF SVM Accuracy:", svm_rbf.score(X_test, y_test))

# Cross-validation scores
linear_cv = cross_val_score(svm_linear, X_pca, y, cv=5)
rbf_cv = cross_val_score(svm_rbf, X_pca, y, cv=5)
print("Linear SVM CV Score:", linear_cv.mean())
print("RBF SVM CV Score:", rbf_cv.mean())

# Visualization function
def plot_decision_boundary(model, title):
    h = .02
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.coolwarm)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(title)
    plt.show()

# Plot decision boundaries
plot_decision_boundary(svm_linear, "SVM with Linear Kernel")
plot_decision_boundary(svm_rbf, "SVM with RBF Kernel")
