import os
import numpy as np
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


def train_and_save_model():
    # 1. 配置路径
    TRAIN_NPY = "../outputs/embeddings_data/train_embeddings.npy"
    TEST_NPY = "../outputs/embeddings_data/test_embeddings.npy"
    TRAIN_PARQUET = "../datasets/promptInjection/train.parquet"
    TEST_PARQUET = "../datasets/promptInjection/test.parquet"
    MODEL_SAVE_PATH = "./models/bge_m3_lr.joblib"

    # 确保目录存在
    os.makedirs("./models", exist_ok=True)

    # 2. 加载数据
    print("正在加载真实数据集...")
    if not all(os.path.exists(f) for f in [TRAIN_NPY, TEST_NPY, TRAIN_PARQUET, TEST_PARQUET]):
        raise FileNotFoundError("请确保 train/test 的 .npy 和 .parquet 文件存在于当前目录。")

    X_train = np.load(TRAIN_NPY)
    y_train = pd.read_parquet(TRAIN_PARQUET)["label"].values

    X_test = np.load(TEST_NPY)
    y_test = pd.read_parquet(TEST_PARQUET)["label"].values

    print(f"数据加载完成: Train={X_train.shape}, Test={X_test.shape}")

    # 3. 初始化 BGE-M3 模型
    # 注意：BGE-M3 维度通常是 1024，如果你之前用的是 MiniLM (384)，请修改 model_name
    model_name = "BAAI/bge-m3"
    print(f"正在加载模型: {model_name} ...")
    encoder = SentenceTransformer(model_name)

    # 4. 【关键步骤】重新生成带归一化的 Embedding
    # 科研严谨性要求：训练和预测必须使用完全相同的预处理逻辑
    print("正在生成带 L2 归一化的 Embedding...")

    # 对训练集进行编码并归一化
    X_train_norm = encoder.encode(
        pd.read_parquet(TRAIN_PARQUET)["text"].astype(str).fillna("").tolist(),
        batch_size=32,            # 可根据显存/内存大小调整
        show_progress_bar=True,
        convert_to_numpy=True     # 直接输出 numpy 数组格式
    )

    # 对测试集进行编码（用于验证模型效果）
    X_test_norm = encoder.encode(
        pd.read_parquet(TEST_PARQUET)["text"].astype(str).fillna("").tolist(),
        batch_size=32,            # 可根据显存/内存大小调整
        show_progress_bar=True,
        convert_to_numpy=True     # 直接输出 numpy 数组格式
    )

    # 5. 训练 Logistic Regression
    print("正在训练 Logistic Regression...")
    lr_clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr_clf.fit(X_train_norm, y_train)

    # 6. 评估模型
    y_pred = lr_clf.predict(X_test_norm)
    # acc = accuracy_score(y_test, y_pred)
    # # 使用 macro 平均计算多分类/二分类通用的指标
    # prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    # rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    # f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print("\n模型评估报告 (Test Set):")
    print(classification_report(y_test, y_pred, target_names=["Safe", "Attack"]))

    # 7. 保存模型
    # 我们将 encoder 和 lr_clf 打包在一起保存，方便后续调用
    model_bundle = {
        "encoder": encoder,
        "classifier": lr_clf,
        "model_name": model_name
    }

    joblib.dump(model_bundle, MODEL_SAVE_PATH)
    print(f"模型已保存至: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_and_save_model()


'''
detail.csv：每条文本的 OOF 决策分数、Attack 倾向值和预测标签。
summary.csv：Question、Instruction、Injection 三类的平均分、标准差、最小值、最大值等。
'''