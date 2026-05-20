# Project-2:

# Data Classification using AI

# Algorithm: KNN(K-Nearest neighbor) on iris dataset

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

#----------------------
# Load and understand the iris dataset
#----------------------

iris=load_iris()
df=pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print()
print('Dataset overview: ')
print(f'Number of samples: {df.shape[0]}')
print(f'Number of features: {df.shape[1]-1}')
print(f'Classes: {df["species"].unique()}')
print()
print('First 5 rows of the dataset:')
print(df.head())
print()
print('Class distribution:')
print(df['species'].value_counts())
print()

print('No of missing values in the dataset: ',df.isnull().sum().sum())

#----------------------
# Step-2 Feature Scaling
#----------------------

X= iris.data
y= iris.target
scalar=StandardScaler()
X_scaled=scalar.fit_transform(X)
print('Feature scaling completed, Mean=0 and Std=1 for each feature.')

#----------------------
# Step-3 Train-test split
#----------------------

X_train, X_test, y_train, y_test =train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=True
)

print('Train-test split completed. Training samples: {}, Testing samples: {}'.format(X_train.shape[0], X_test.shape[0]))

#----------------------
# Find value of K using elbow method
#----------------------

error_rates=[]
k_values=range(1, 21)

for k in k_values:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    preds=knn.predict(X_test)
    error_rates.append(1-accuracy_score(y_test, preds))

optimal_k=k_values[np.argmin(error_rates)]
print('Optimal K value found using elbow method: {}'.format(optimal_k))

#----------------------
# Step-4 Train KNN model with optimal K
#----------------------
knn=KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train, y_train)
predictions=knn.predict(X_test)
print('Model training completed with K={}. Predictions on test set done.'.format(optimal_k))

#----------------------
# Step-5 Evaluate the model
#----------------------
acc=accuracy_score(y_test, predictions)
f1=f1_score(y_test, predictions, average='weighted')
cm=confusion_matrix(y_test, predictions)

print('*' * 50)
print('Model Evaluation Results:')
print('Accuracy: {:.2f}%'.format(acc*100))
print('F1 Score: {:.2f}'.format(f1))
print()
print('Classification Report:')
print(classification_report(y_test, predictions, target_names=iris.target_names))
print('Confusion Matrix:')
print(cm)

#----------------------
# Step-6 Visualize the results
#----------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Project 2 — Data Classification Using AI (KNN on Iris)\nDecodeLabs | Batch 2026",
             fontsize=13, fontweight='bold', color='#1a3a5c')
 
# Plot 1: Elbow Curve
axes[0].plot(k_values, error_rates, color='#1a3a5c', marker='o', linewidth=2, markersize=6)
axes[0].axvline(x=optimal_k, color='#e05a00', linestyle='--', linewidth=2,
                label=f'Optimal K={optimal_k}')
axes[0].scatter([optimal_k], [error_rates[optimal_k - 1]],
                color='#e05a00', s=150, zorder=5)
axes[0].set_title('Elbow Method: Choosing K', fontweight='bold')
axes[0].set_xlabel('K Value')
axes[0].set_ylabel('Error Rate')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
 
# Plot 2: Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names,
            ax=axes[1], linewidths=0.5)
axes[1].set_title('Confusion Matrix', fontweight='bold')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')
 
# Plot 3: Feature Scatter (Petal Length vs Petal Width)
colors = ['#1a3a5c', '#e05a00', '#2e8b57']
for idx, species in enumerate(iris.target_names):
    mask = y == idx
    axes[2].scatter(X[mask, 2], X[mask, 3],
                    label=species, color=colors[idx],
                    alpha=0.7, edgecolors='white', s=60)
 
axes[2].set_title('Feature Space: Petal L vs Petal W', fontweight='bold')
axes[2].set_xlabel('Petal Length (cm)')
axes[2].set_ylabel('Petal Width (cm)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
 
plt.tight_layout()
plt.show()
print("\n📊 Visualization saved!")

#----------------------
# Predict on new data
#----------------------

new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example: [sepal length, sepal width, petal length, petal width]
new_sample_scaled = scalar.transform(new_sample)
predicted_class = knn.predict(new_sample_scaled)
print('\nPrediction for new sample {}: {}'.format(new_sample[0], iris.target_names[predicted_class][0]))


print('\nProject 2 completed successfully! KNN model trained and evaluated on the Iris dataset.')
# End of code
