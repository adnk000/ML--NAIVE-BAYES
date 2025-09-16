import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Load the datasets
train_100D = pd.read_csv('100D_heart_disease_dataset.csv')
train_300D = pd.read_csv('300D_heart_disease_dataset.csv')
train_600D = pd.read_csv('600D_heart_disease_dataset.csv')
train_900D = pd.read_csv('900D_heart_disease_dataset.csv')
test_data = pd.read_csv('100D_test_dataset.csv')

# Prepare the data
X_test = test_data.drop('target', axis=1)
y_test = test_data['target']

datasets = {
    "100D": train_100D,
    "300D": train_300D,
    "600D": train_600D,
    "900D": train_900D
}

results = {}

# Train and evaluate the Naive Bayes model for each dataset
for name, df in datasets.items():
    X_train = df.drop('target', axis=1)
    y_train = df['target']

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    results[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc
    }

# --- Plotting ---

# 1. Plot Accuracy, Precision, and Recall
metrics_df = pd.DataFrame({
    'Accuracy': [results[name]['accuracy'] for name in datasets.keys()],
    'Precision': [results[name]['precision'] for name in datasets.keys()],
    'Recall': [results[name]['recall'] for name in datasets.keys()]
}, index=datasets.keys())

metrics_df.plot(kind='bar', figsize=(12, 7))
plt.title('Comparison of Metrics for Different Sized Datasets')
plt.ylabel('Score')
plt.xlabel('Dataset Size')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('metrics_comparison.png')
plt.close()

# 2. Plot AUC/ROC Curve
plt.figure(figsize=(10, 8))
for name, result in results.items():
    plt.plot(result['fpr'], result['tpr'],
             label=f'{name} (AUC = {result["roc_auc"]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig('roc_curve_comparison.png')
plt.close()

print("Results:")
for name, metrics in results.items():
    print(f"\nDataset: {name}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  AUC: {metrics['roc_auc']:.4f}")