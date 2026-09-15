# Machine Learning Algorithms From Scratch 🧠⚙️

A comprehensive collection of foundational machine learning algorithms implemented purely in **Python** and **NumPy**. 

As a Data Scientist and AI Engineer, I built this repository to move beyond black-box libraries like Scikit-Learn and deeply understand the mathematical engines powering these models. Every algorithm here is built from the ground up, focusing on vectorized operations, computational efficiency, and clean architecture.

## 🚀 Why This Project?
While modern frameworks are great for production, building algorithms from scratch is the ultimate test of understanding. This project demonstrates:
* **Mathematical Translation:** Converting complex calculus and linear algebra into functional code.
* **Algorithmic Efficiency:** Utilizing NumPy vectorization to avoid slow loops.
* **Core ML Concepts:** Hands-on implementation of gradient descent, loss functions, kernel tricks, and entropy calculations.

---

## 🧰 Algorithms Implemented

### Supervised Learning
**Regression:**
- [ ] Linear Regression (Ordinary Least Squares & Gradient Descent)
- [ ] Ridge Regression (L2 Regularization)
- [ ] Lasso Regression (L1 Regularization)

**Classification:**
- [ ] Logistic Regression 
- [ ] K-Nearest Neighbors (KNN)
- [ ] Support Vector Machines (SVM) with Linear & RBF Kernels
- [ ] Naive Bayes (Gaussian & Multinomial)
- [ ] Decision Trees (CART algorithm, Gini Impurity & Entropy)
- [ ] Random Forest

### Unsupervised Learning
**Clustering:**
- [ ] K-Means Clustering
- [ ] DBSCAN (Density-Based Spatial Clustering)
- [ ] Hierarchical Clustering

**Dimensionality Reduction:**
- [ ] Principal Component Analysis (PCA)

---

## 📁 Repository Structure

Each algorithm is contained within its own dedicated module, complete with a clean class structure (mimicking the `.fit()` and `.predict()` API) and a test script to visualize its performance on sample datasets.

```text
├── supervised_learning/
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   └── ...
├── unsupervised_learning/
│   ├── k_means.py
│   ├── pca.py
│   └── ...
├── utils/
│   ├── data_manipulation.py (train_test_split, scaling, etc.)
│   ├── metrics.py (accuracy, MSE, F1-score, etc.)
│   └── ...
├── notebooks/ (Jupyter notebooks with visual explorations)
├── requirements.txt
└── README.md
