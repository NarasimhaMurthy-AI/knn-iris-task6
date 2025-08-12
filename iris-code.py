# knn_task6.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

# Load dataset
df = pd.read_csv("Iris.csv")
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

X = df.drop(columns=['Species'])
y = df['Species']

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.3, random_state=42, stratify=y_enc)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k_values = [1,3,5,7,9,11]
results = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    preds = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    results.append({'k': k, 'accuracy': acc, 'confusion_matrix': cm})
    print(f"K={k} -> Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", cm, "\n")

# Plot accuracy vs k
import pandas as pd
results_df = pd.DataFrame([{'k': r['k'], 'accuracy': r['accuracy']} for r in results])
plt.plot(results_df['k'], results_df['accuracy'], marker='o')
plt.title("K value vs Accuracy (Iris dataset)")
plt.xlabel("Number of neighbors (k)")
plt.ylabel("Accuracy on test set")
plt.grid(True)
plt.show()

# Decision boundary for two features
f1, f2 = 'PetalLengthCm', 'PetalWidthCm'
X2 = df[[f1, f2]]
y2 = le.transform(df['Species'])
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=42, stratify=y2)
scaler2 = StandardScaler()
X2_train_s = scaler2.fit_transform(X2_train)
X2_test_s = scaler2.transform(X2_test)

best = max(results, key=lambda r: r['accuracy'])
best_k = best['k']
knn2 = KNeighborsClassifier(n_neighbors=best_k)
knn2.fit(X2_train_s, y2_train)
DecisionBoundaryDisplay.from_estimator(knn2, X2_train_s)
plt.scatter(X2_train_s[:,0], X2_train_s[:,1], c=y2_train, edgecolor='k')
plt.title(f"Decision boundary (k={best_k}) on scaled features: {f1} vs {f2}")
plt.show()

# Final confusion matrix and report on full features for best k
best_knn_full = KNeighborsClassifier(n_neighbors=best_k)
best_knn_full.fit(X_train_scaled, y_train)
preds_full = best_knn_full.predict(X_test_scaled)
print(confusion_matrix(y_test, preds_full))
print(classification_report(y_test, preds_full, target_names=le.classes_))

# ...existing code...

# Create DataFrame for cleaned/scaled test features
X_test_cleaned = pd.DataFrame(X_test_scaled, columns=X.columns)

# Add actual and predicted labels
X_test_cleaned['Actual'] = le.inverse_transform(y_test)
X_test_cleaned['Predicted'] = le.inverse_transform(preds_full)

# Save to CSV
X_test_cleaned.to_csv(r"D:\AIML PROJECTS\iris_test_results.csv", index=False)

print("File saved as iris_test_results.csv")

# ...existing code...