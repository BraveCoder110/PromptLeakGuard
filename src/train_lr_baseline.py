'''
Logistic Regression Baseline
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def train_and_evaluate_lr(train_npy_path, test_npy_path, train_parquet_path, test_parquet_path, label_column="label"):
    """
    Load embeddings and labels, train a logistic regression model, and output comprehensive evaluation metrics.
    """
    import pandas as pd  # Defer imports to prevent affecting other functions when not installed.

    # 1. loading
    print("loading Embedding and label data...")
    try:
        X_train = np.load(train_npy_path)
        X_test = np.load(test_npy_path)

        y_train = pd.read_parquet(train_parquet_path)[label_column].values
        y_test = pd.read_parquet(test_parquet_path)[label_column].values
    except Exception as e:
        print(f"failed: {e}")
        return

    print(f"training shape: {X_train.shape}, test shape: {X_test.shape}")

    # 2. train Logistic Regression
    print("training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    print("train done！")

    # 3. predict
    print("predicting...")
    y_pred = model.predict(X_test)

    # 4. Calculate the various evaluation metrics.
    acc = accuracy_score(y_test, y_pred)
    # Calculate metrics applicable to both multi-class and binary classification using macro-averaging.
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print("\n" + "="*40)
    print("Model evaluation metrics (Test Set)")
    print("="*40)
    print(f"🎯 Accuracy:  {acc:.4f}")
    print(f"🎯 Precision: {prec:.4f}")
    print(f"🎯 Recall:    {rec:.4f}")
    print(f"🎯 F1-score:  {f1:.4f}")
    print("="*40)

    # 5. Plot the confusion matrix.
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Logistic Regression Confusion Matrix")
    plt.tight_layout()

    output_image = "../outputs/logistic_regression_analysis/lr_confusion_matrix.png"
    plt.savefig(output_image, dpi=150)
    print(f"\nThe confusion matrix has been saved as: {output_image}")
    plt.show()

if __name__ == "__main__":
    # Ensure that your Parquet file contains the corresponding label column.
    LABEL_COL = "label"

    train_and_evaluate_lr(
        train_npy_path="../outputs/embeddings_data/train_embeddings.npy",
        test_npy_path="../outputs/embeddings_data/test_embeddings.npy",
        train_parquet_path="../datasets/promptInjection/train.parquet",
        test_parquet_path="../datasets/promptInjection/test.parquet",
        label_column=LABEL_COL
    )



'''
Unified Data Source: Simultaneously reads `.npy` (features) and `.parquet` (labels) files, ensuring indices are perfectly aligned to prevent misalignment.
Robust Metric Calculation: Uses `average="macro"` for Precision, Recall, and F1 scores, ensuring correct calculation regardless of whether labels are numeric (0/1) or strings; includes `zero_division=0` to prevent errors in edge cases.
Confusion Matrix Visualization: Uses `sklearn`'s built-in `ConfusionMatrixDisplay` to generate an intuitive confusion matrix and automatically saves it as an image.
Model Stability: Sets `max_iter=1000` in `LogisticRegression` to prevent convergence warnings caused by 384-dimensional features, and sets `random_state=42` to ensure reproducible results across runs.
'''