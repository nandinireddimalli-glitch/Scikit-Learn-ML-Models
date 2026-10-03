from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

print("Dataset loaded successfully!")
print("Number of samples:", data.data.shape[0])
print("Number of features:", data.data.shape[1])
print("Classes:", data.target_names)

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# 1. Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset loaded successfully!")
print("Features:", X.shape)
print("Target:", y.shape)


# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nData split completed!")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# 3. Scale the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 4. Create Logistic Regression model
model = LogisticRegression(max_iter=1000)


# 5. Train the model
model.fit(X_train, y_train)

print("\nLogistic Regression model trained successfully!")


# 6. Make predictions
y_pred = model.predict(X_test)


# 7. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Evaluation")
print("----------------")
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=data.target_names
))

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
# 8. Create Decision Tree model
tree_model = DecisionTreeClassifier(
    random_state=42
)

# 9. Train the Decision Tree
tree_model.fit(X_train, y_train)

print("\nDecision Tree model trained successfully!")

# 10. Make predictions
tree_pred = tree_model.predict(X_test)

# 11. Evaluate Decision Tree
tree_accuracy = accuracy_score(y_test, tree_pred)

print("\nDecision Tree Evaluation")
print("------------------------")
print("Accuracy:", tree_accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    tree_pred,
    target_names=data.target_names
))
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
# 12. Create Random Forest model
forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 13. Train the Random Forest
forest_model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")

# 14. Make predictions
forest_pred = forest_model.predict(X_test)

# 15. Evaluate Random Forest
forest_accuracy = accuracy_score(y_test, forest_pred)

print("\nRandom Forest Evaluation")
print("------------------------")
print("Accuracy:", forest_accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    forest_pred,
    target_names=data.target_names
))
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
# 16. Create KNN model
knn_model = KNeighborsClassifier(
    n_neighbors=5
)

# 17. Train the KNN model
knn_model.fit(X_train, y_train)

print("\nKNN model trained successfully!")

# 18. Make predictions
knn_pred = knn_model.predict(X_test)

# 19. Evaluate KNN
knn_accuracy = accuracy_score(y_test, knn_pred)

print("\nKNN Evaluation")
print("--------------")
print("Accuracy:", knn_accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    knn_pred,
    target_names=data.target_names
))
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
# ==========================================
# LINEAR REGRESSION
# ==========================================

# 20. Load diabetes dataset
diabetes = load_diabetes()

X_reg = diabetes.data
y_reg = diabetes.target

print("\n\nRegression Dataset")
print("------------------")
print("Features:", X_reg.shape)
print("Target:", y_reg.shape)
# 21. Split regression data
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

print("Training samples:", X_reg_train.shape[0])
print("Testing samples:", X_reg_test.shape[0])
# 22. Create Linear Regression model
linear_model = LinearRegression()

# 23. Train the model
linear_model.fit(X_reg_train, y_reg_train)

print("\nLinear Regression model trained successfully!")
# 24. Make predictions
y_reg_pred = linear_model.predict(X_reg_test)
# 25. Calculate regression metrics
mae = mean_absolute_error(y_reg_test, y_reg_pred)
mse = mean_squared_error(y_reg_test, y_reg_pred)
r2 = r2_score(y_reg_test, y_reg_pred)

print("\nLinear Regression Evaluation")
print("----------------------------")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
# ==========================================
# CLASSIFICATION MODEL COMPARISON
# ==========================================

# Calculate metrics for each classification model

logistic_metrics = [
    accuracy_score(y_test, y_pred),
    precision_score(y_test, y_pred),
    recall_score(y_test, y_pred),
    f1_score(y_test, y_pred)
]

tree_metrics = [
    accuracy_score(y_test, tree_pred),
    precision_score(y_test, tree_pred),
    recall_score(y_test, tree_pred),
    f1_score(y_test, tree_pred)
]

forest_metrics = [
    accuracy_score(y_test, forest_pred),
    precision_score(y_test, forest_pred),
    recall_score(y_test, forest_pred),
    f1_score(y_test, forest_pred)
]

knn_metrics = [
    accuracy_score(y_test, knn_pred),
    precision_score(y_test, knn_pred),
    recall_score(y_test, knn_pred),
    f1_score(y_test, knn_pred)
]

print("\n\nCLASSIFICATION MODEL COMPARISON")
print("================================")

print(
    f"{'Model':<22}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1-Score':<12}"
)

print("-" * 68)

print(
    f"{'Logistic Regression':<22}"
    f"{logistic_metrics[0]:<12.4f}"
    f"{logistic_metrics[1]:<12.4f}"
    f"{logistic_metrics[2]:<12.4f}"
    f"{logistic_metrics[3]:<12.4f}"
)

print(
    f"{'Decision Tree':<22}"
    f"{tree_metrics[0]:<12.4f}"
    f"{tree_metrics[1]:<12.4f}"
    f"{tree_metrics[2]:<12.4f}"
    f"{tree_metrics[3]:<12.4f}"
)

print(
    f"{'Random Forest':<22}"
    f"{forest_metrics[0]:<12.4f}"
    f"{forest_metrics[1]:<12.4f}"
    f"{forest_metrics[2]:<12.4f}"
    f"{forest_metrics[3]:<12.4f}"
)

print(
    f"{'KNN':<22}"
    f"{knn_metrics[0]:<12.4f}"
    f"{knn_metrics[1]:<12.4f}"
    f"{knn_metrics[2]:<12.4f}"
    f"{knn_metrics[3]:<12.4f}"
)
import matplotlib.pyplot as plt
# ==========================================
# VISUALIZE MODEL PERFORMANCE
# ==========================================

models = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "KNN"
]

accuracies = [
    logistic_metrics[0],
    tree_metrics[0],
    forest_metrics[0],
    knn_metrics[0]
]

plt.figure(figsize=(10, 6))

plt.bar(models, accuracies)

plt.title("Classification Model Accuracy Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy")

plt.ylim(0, 1)

plt.xticks(rotation=20)

plt.tight_layout()

plt.show()