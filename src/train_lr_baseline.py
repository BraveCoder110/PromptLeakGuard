'''
Logistic Regression Baseline
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def train_and_evaluate_lr(train_npy_path, test_npy_path, train_parquet_path, test_parquet_path, label_column="label"):
    """
    加载 Embedding 和标签，训练 Logistic Regression 并输出完整评估指标。
    """
    import pandas as pd  # 延迟导入，防止未安装时影响其他功能

    # 1. 加载数据
    print("🔄 正在加载 Embedding 和标签数据...")
    try:
        X_train = np.load(train_npy_path)
        X_test = np.load(test_npy_path)

        # 从 parquet 中提取真实的标签
        y_train = pd.read_parquet(train_parquet_path)[label_column].values
        y_test = pd.read_parquet(test_parquet_path)[label_column].values
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    print(f"✅ 训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")

    # 2. 训练 Logistic Regression
    print("🤖 正在训练 Logistic Regression 模型...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    print("✅ 模型训练完成！")

    # 3. 预测
    print("🔮 正在生成预测结果...")
    y_pred = model.predict(X_test)

    # 4. 计算各项评估指标
    acc = accuracy_score(y_test, y_pred)
    # 使用 macro 平均计算多分类/二分类通用的指标
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print("\n" + "="*40)
    print("📊 模型评估指标 (Test Set)")
    print("="*40)
    print(f"🎯 Accuracy:  {acc:.4f}")
    print(f"🎯 Precision: {prec:.4f}")
    print(f"🎯 Recall:    {rec:.4f}")
    print(f"🎯 F1-score:  {f1:.4f}")
    print("="*40)

    # 5. 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Logistic Regression Confusion Matrix")
    plt.tight_layout()

    output_image = "../outputs/logistic_regression_analysis/lr_confusion_matrix.png"
    plt.savefig(output_image, dpi=150)
    print(f"\n📊 混淆矩阵已保存为: {output_image}")
    plt.show()

if __name__ == "__main__":
    # 确保你的 parquet 文件中有对应的标签列
    LABEL_COL = "label"

    train_and_evaluate_lr(
        train_npy_path="../outputs/embeddings_data/train_embeddings.npy",
        test_npy_path="../outputs/embeddings_data/test_embeddings.npy",
        train_parquet_path="../datasets/promptInjection/train.parquet",
        test_parquet_path="../datasets/promptInjection/test.parquet",
        label_column=LABEL_COL
    )



'''
统一数据源：通过同时读取 .npy（特征）和 .parquet（标签），确保特征和标签的索引完全对齐，避免错位。
鲁棒的指标计算：使用了 average="macro" 来计算 Precision、Recall 和 F1，这样无论你的标签是数字（0/1）还是字符串，都能正确计算。同时加入了 zero_division=0 防止极端情况报错。
可视化混淆矩阵：利用 sklearn 内置的 ConfusionMatrixDisplay 绘制了直观的混淆矩阵，并自动保存为图片。
模型稳定性：在 LogisticRegression 中设置了 max_iter=1000，防止 384 维特征导致模型不收敛而抛出警告；设置 random_state=42 保证每次运行结果可复现。
运行这段代码后，你将在控制台看到清晰的评估指标，并在当前目录下得到一张混淆矩阵图片。
'''