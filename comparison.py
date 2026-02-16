"""
The model I chose was a decision tree. Decision trees work by repeatedly
splitting the data to reduce impurity in each group. Impurity measures how
mixed the values in a node are: for classification, a node has low impurity
if most samples belong to the same class, and high impurity if the classes
are evenly mixed. For regression, impurity instead reflects how spread out
the numeric values are, so lower variance means lower impurity. Since this
task is binary classification, I used entropy as the impurity measure,
given by H = -sum(p_k log2(p_k)), where p_k is the proportion of each class.
I chose entropy because I previously studied information theory and was
interested in seeing it used here to measure uncertainty in the data.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from binary_classification import load_data


# Load dataset using your existing loader
X_train, X_test, y_train, y_test, _ = load_data()

# Convert torch tensors -> numpy for sklearn
X_train = X_train.numpy()
X_test = X_test.numpy()
y_train = y_train.numpy()
y_test = y_test.numpy()


# Train sklearn decision tree 
clf = DecisionTreeClassifier(random_state=42, criterion= "entropy")
clf.fit(X_train, y_train)


# Compute test accuracy of tree
tree_test_acc = np.mean(clf.predict(X_test) == y_test)

# From-scratch logistic regression result 
logistic_test_acc = 0.9912


print("Model Comparison")
print("----------------")
print(f"From-scratch Logistic Regression Test Accuracy: {logistic_test_acc:.4f}")
print(f"Sklearn Decision Tree Test Accuracy:            {tree_test_acc:.4f}")


"""
The logistic regression performed better than the decision tree, this may be because the decision trees have a strong tendency to overfitt
which may lead it to perform worse on actual predictions, an ensemble tree or gradeint boosted tree may perform better.


"""
