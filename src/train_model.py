import os
import numpy as np
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


def train_and_save_model():
    # 1. file paths
    TRAIN_NPY = "../outputs/embeddings_data/train_embeddings.npy"
    TEST_NPY = "../outputs/embeddings_data/test_embeddings.npy"
    TRAIN_PARQUET = "../datasets/promptInjection/train.parquet"
    TEST_PARQUET = "../datasets/promptInjection/test.parquet"
    MODEL_SAVE_PATH = "./models/bge_m3_lr.joblib"

    # check dir
    os.makedirs("./models", exist_ok=True)

    # 2. loading
    print("loading data...")
    if not all(os.path.exists(f) for f in [TRAIN_NPY, TEST_NPY, TRAIN_PARQUET, TEST_PARQUET]):
        raise FileNotFoundError("Please ensure that the train/test .npy and .parquet files exist in the current directory.")

    X_train = np.load(TRAIN_NPY)
    y_train = pd.read_parquet(TRAIN_PARQUET)["label"].values

    X_test = np.load(TEST_NPY)
    y_test = pd.read_parquet(TEST_PARQUET)["label"].values

    print(f"Data loading complete.: Train={X_train.shape}, Test={X_test.shape}")

    # 3. Initialize the BGE-M3 model
    # Note: The BGE-M3 dimension is typically 1024; if you previously used MiniLM (384), please update the model_name.
    model_name = "BAAI/bge-m3"
    print(f"Loading model...: {model_name} ...")
    encoder = SentenceTransformer(model_name)

    # 4. [Critical Step] Regenerate normalized embeddings
    # Requirement for scientific rigor: Training and prediction must use identical preprocessing logic.
    print("Generating embeddings with L2 normalization...")

    # Encode and normalize the training set.
    X_train_norm = encoder.encode(
        pd.read_parquet(TRAIN_PARQUET)["text"].astype(str).fillna("").tolist(),
        batch_size=32,            # Can be adjusted based on VRAM/RAM size.
        show_progress_bar=True,
        convert_to_numpy=True     # Directly output in NumPy array format.
    )

    # Encode the test set (used to evaluate model performance)
    X_test_norm = encoder.encode(
        pd.read_parquet(TEST_PARQUET)["text"].astype(str).fillna("").tolist(),
        batch_size=32,            # Can be adjusted based on VRAM/RAM size.
        show_progress_bar=True,
        convert_to_numpy=True     # Directly output in NumPy array format.
    )

    # 5. train Logistic Regression
    print("train Logistic Regression...")
    lr_clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr_clf.fit(X_train_norm, y_train)

    # 6. Evaluation model
    y_pred = lr_clf.predict(X_test_norm)
    # acc = accuracy_score(y_test, y_pred)
    # # Calculate metrics applicable to both multi-class and binary classification using macro-averaging
    # prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    # rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    # f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print("\nreport (Test Set):")
    print(classification_report(y_test, y_pred, target_names=["Safe", "Attack"]))

    # 7. save model
    model_bundle = {
        "encoder": encoder,
        "classifier": lr_clf,
        "model_name": model_name
    }

    joblib.dump(model_bundle, MODEL_SAVE_PATH)
    print(f"model is saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_and_save_model()


'''
detail.csv：每条文本的 OOF 决策分数、Attack 倾向值和预测标签。
summary.csv：Question、Instruction、Injection 三类的平均分、标准差、最小值、最大值等。
'''