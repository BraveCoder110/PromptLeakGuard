"""
Day18：Logistic Regression 分类器决策分析。

功能：
1. 读取已经保存的 BGE-M3 Embedding；
2. 使用 expected_label 训练 Safe vs Attack 二分类 Logistic Regression；
3. 使用 GroupKFold 生成逐条文本的 OOF decision score；
4. 比较 Question、Instruction、Injection 三类 Prompt 的平均决策分数；
5. 保存逐条结果、分类汇总结果和最终分类器模型。

决策分数解释：
- Attack 被编码为正类 1，Safe 被编码为负类 0；
- decision score 越大，模型越倾向于判断为 Attack / Injection；
- decision score 小于 0，模型更倾向于判断为 Safe；
- decision score 不是概率，但可用 sigmoid 转换为未校准的 Attack 倾向值。
"""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# =========================
# 1. 路径及字段配置
# =========================
# 脚本放在项目代码目录时，可按项目结构调整 BASE_DIR。
SCRIPT_DIR = Path(__file__).resolve().parent

# 优先支持当前上传文件目录；也兼容原项目中的 ../outputs 路径。
CANDIDATE_EMBEDDING_PATHS = [
    SCRIPT_DIR / "day17_embeddings_bgeM3.npy",
    SCRIPT_DIR / "outputs" / "embeddings_data" / "day17_embeddings_bgeM3.npy",
    SCRIPT_DIR.parent / "outputs" / "embeddings_data" / "day17_embeddings_bgeM3.npy",
]
CANDIDATE_INDEX_PATHS = [
    SCRIPT_DIR / "day17_embeddings_bgeM3_index.csv",
    SCRIPT_DIR / "outputs" / "embeddings_data" / "day17_embeddings_bgeM3_index.csv",
    SCRIPT_DIR.parent / "outputs" / "embeddings_data" / "day17_embeddings_bgeM3_index.csv",
]

OUTPUT_DIR = SCRIPT_DIR / "classifier_decision_outputs"
OUTPUT_DETAIL_PATH = OUTPUT_DIR / "day17_classifier_decision_detail_bgeM3.csv"
OUTPUT_SUMMARY_PATH = OUTPUT_DIR / "day17_classifier_decision_summary_bgeM3.csv"
OUTPUT_MODEL_PATH = OUTPUT_DIR / "day17_logistic_regression_bgeM3.joblib"

GROUP_COLUMN = "group_id"
PROMPT_TYPE_COLUMN = "prompt_type"
LABEL_COLUMN = "expected_label"
REQUIRED_PROMPT_TYPES = ["Question", "Instruction", "Injection"]
NEGATIVE_LABEL = "Safe"
POSITIVE_LABEL = "Attack"
RANDOM_STATE = 42


def find_existing_path(candidates: list[Path], description: str) -> Path:
    """从候选路径中找到第一个实际存在的文件。"""
    for path in candidates:
        if path.exists():
            return path

    print(f"未找到{description}，已检查以下路径：")
    for path in candidates:
        print(f"  - {path}")
    sys.exit(1)


# =========================
# 2. 加载 Embedding 和索引表
# =========================
embedding_path = find_existing_path(
    CANDIDATE_EMBEDDING_PATHS,
    "Embedding 文件",
)
index_path = find_existing_path(
    CANDIDATE_INDEX_PATHS,
    "Embedding 索引 CSV",
)

print(f"正在读取 Embedding：{embedding_path}")
embeddings = np.load(embedding_path)
embeddings = np.asarray(embeddings, dtype=np.float32)

print(f"正在读取索引数据：{index_path}")
try:
    index_df = pd.read_csv(index_path, encoding="utf-8-sig")
except UnicodeDecodeError:
    index_df = pd.read_csv(index_path, encoding="gb18030")

required_columns = {
    GROUP_COLUMN,
    PROMPT_TYPE_COLUMN,
    LABEL_COLUMN,
}
missing_columns = required_columns - set(index_df.columns)
if missing_columns:
    print(f"索引 CSV 缺少必要字段：{sorted(missing_columns)}")
    sys.exit(1)

if embeddings.ndim != 2:
    print(f"Embedding 应为二维数组，实际形状为：{embeddings.shape}")
    sys.exit(1)

if len(index_df) != embeddings.shape[0]:
    print(
        "Embedding 数量与索引 CSV 行数不一致："
        f"Embedding={embeddings.shape[0]}，CSV={len(index_df)}"
    )
    sys.exit(1)

# 如果索引文件带有 embedding_index，检查它是否与 .npy 行顺序一致。
if "embedding_index" in index_df.columns:
    expected_indexes = np.arange(len(index_df))
    actual_indexes = index_df["embedding_index"].to_numpy()
    if not np.array_equal(actual_indexes, expected_indexes):
        print("embedding_index 不是从 0 开始的连续行号，无法保证向量映射正确。")
        sys.exit(1)

# 清理字符串字段，避免空格造成类别匹配失败。
index_df[PROMPT_TYPE_COLUMN] = (
    index_df[PROMPT_TYPE_COLUMN].astype(str).str.strip()
)
index_df[LABEL_COLUMN] = index_df[LABEL_COLUMN].astype(str).str.strip()

unknown_prompt_types = sorted(
    set(index_df[PROMPT_TYPE_COLUMN].unique()) - set(REQUIRED_PROMPT_TYPES)
)
if unknown_prompt_types:
    print(f"发现未知 prompt_type：{unknown_prompt_types}")
    sys.exit(1)

unknown_labels = sorted(
    set(index_df[LABEL_COLUMN].unique()) - {NEGATIVE_LABEL, POSITIVE_LABEL}
)
if unknown_labels:
    print(f"发现未知 expected_label：{unknown_labels}")
    sys.exit(1)


# =========================
# 3. 构造二分类目标
# =========================
# Safe -> 0，Attack -> 1。
# 因为 Attack 是正类，所以 decision_function 分数越大越偏向 Attack。
y = (index_df[LABEL_COLUMN] == POSITIVE_LABEL).astype(int).to_numpy()
groups = index_df[GROUP_COLUMN].to_numpy()

if len(np.unique(y)) != 2:
    print("数据必须同时包含 Safe 和 Attack 两类。")
    sys.exit(1)

number_of_groups = index_df[GROUP_COLUMN].nunique()
n_splits = min(5, number_of_groups)
if n_splits < 2:
    print("至少需要两个 group_id 才能进行 GroupKFold。")
    sys.exit(1)


# =========================
# 4. 定义 Logistic Regression 分类器
# =========================
# Embedding 各维度尺度通常相近，但标准化后有利于线性模型稳定训练。
# with_mean=False 不会对矩阵执行中心化，也兼容未来可能使用的稀疏矩阵。
classifier = make_pipeline(
    StandardScaler(with_mean=False),
    LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
)


# =========================
# 5. 生成 OOF decision score
# =========================
# 不直接使用 classifier.fit(...).decision_function(embeddings) 作为报告结果，
# 因为那是训练集内分数，可能过于乐观。
#
# GroupKFold 保证同一个 group_id 的三条 Prompt 总是在同一折：
# Question、Instruction、Injection 不会一部分进入训练集、一部分进入验证集，
# 从而减少同组语义高度相似造成的数据泄漏。
group_cv = GroupKFold(n_splits=n_splits)
print(f"正在使用 {n_splits} 折 GroupKFold 计算 OOF decision score……")

oof_decision_scores = cross_val_predict(
    classifier,
    embeddings,
    y,
    groups=groups,
    cv=group_cv,
    method="decision_function",
)
oof_decision_scores = np.asarray(oof_decision_scores, dtype=float).reshape(-1)

# 将 decision score 通过 sigmoid 转为 0～1 倾向值。
# 该值没有经过概率校准，因此命名为 attack tendency，而不是正式概率。
oof_attack_tendency = 1.0 / (1.0 + np.exp(-oof_decision_scores))


# =========================
# 6. 保存逐条决策结果
# =========================
detail_df = index_df.copy()
if "embedding_index" not in detail_df.columns:
    detail_df.insert(0, "embedding_index", np.arange(len(detail_df)))

detail_df["target_binary"] = y
detail_df["oof_decision_score"] = oof_decision_scores
detail_df["oof_attack_tendency"] = oof_attack_tendency
detail_df["oof_predicted_label"] = np.where(
    oof_decision_scores >= 0,
    POSITIVE_LABEL,
    NEGATIVE_LABEL,
)


# =========================
# 7. 按 Prompt 类型汇总
# =========================
summary_df = (
    detail_df.groupby(PROMPT_TYPE_COLUMN, as_index=False)
    .agg(
        mean_decision_score=("oof_decision_score", "mean"),
        median_decision_score=("oof_decision_score", "median"),
        std_decision_score=("oof_decision_score", "std"),
        min_decision_score=("oof_decision_score", "min"),
        max_decision_score=("oof_decision_score", "max"),
        mean_attack_tendency=("oof_attack_tendency", "mean"),
        sample_count=("oof_decision_score", "size"),
    )
)

# 固定输出顺序，便于直接观察 Question -> Instruction -> Injection 的变化。
prompt_type_order = pd.CategoricalDtype(
    categories=REQUIRED_PROMPT_TYPES,
    ordered=True,
)
summary_df[PROMPT_TYPE_COLUMN] = summary_df[PROMPT_TYPE_COLUMN].astype(
    prompt_type_order
)
summary_df = summary_df.sort_values(PROMPT_TYPE_COLUMN).reset_index(drop=True)
summary_df[PROMPT_TYPE_COLUMN] = summary_df[PROMPT_TYPE_COLUMN].astype(str)


# =========================
# 8. 保存文件和最终模型
# =========================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

detail_df.to_csv(
    OUTPUT_DETAIL_PATH,
    index=False,
    encoding="utf-8-sig",
    float_format="%.6f",
)
summary_df.to_csv(
    OUTPUT_SUMMARY_PATH,
    index=False,
    encoding="utf-8-sig",
    float_format="%.6f",
)

# 使用全部数据训练最终模型，供以后对新文本的 Embedding 调用：
# classifier.decision_function(new_embeddings)
classifier.fit(embeddings, y)
joblib.dump(classifier, OUTPUT_MODEL_PATH)

print("\n三类 Prompt 的分类器决策汇总：")
print(
    summary_df[
        [
            PROMPT_TYPE_COLUMN,
            "mean_decision_score",
            "std_decision_score",
            "mean_attack_tendency",
            "sample_count",
        ]
    ].to_string(index=False)
)

print("\n结果解释：")
print("- mean_decision_score 越高，越偏向 Attack / Injection。")
print("- 预期通常是 Question 较低、Instruction 居中、Injection 较高。")
print("- 但最终排序必须以实际计算结果为准，不能预先假定。")

print("\n已保存：")
print(f"- 逐条结果：{OUTPUT_DETAIL_PATH}")
print(f"- 分类汇总：{OUTPUT_SUMMARY_PATH}")
print(f"- 最终模型：{OUTPUT_MODEL_PATH}")
