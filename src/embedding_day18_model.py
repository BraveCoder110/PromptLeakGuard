"""
生成 Embedding，并完成 Day18 任务三：Embedding 空间检查。

输入数据来自 Day17：../docs/day17_instruction_bias.csv
每个 group_id 应包含三条 Prompt：Question、Instruction、Injection。

程序会输出：
1. 所有文本逐条对应的 Embedding：.npy 文件
2. Embedding 与原始数据的对应关系：CSV 文件
3. 每组三类 Prompt 两两之间的余弦相似度：CSV 文件
"""

import os
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# =========================
# 1. 配置参数
# =========================
MODEL_NAME = "BAAI/bge-m3"
MODEL_FILE_NAME = "bgeM3"

INPUT_CSV_PATH = "../docs/day17_instruction_bias.csv"
OUTPUT_EMBEDDING_PATH = (
    "../outputs/embeddings_data/"
    f"day17_embeddings_{MODEL_FILE_NAME}.npy"
)
OUTPUT_EMBEDDING_INDEX_PATH = (
    "../outputs/embeddings_data/"
    f"day17_embeddings_{MODEL_FILE_NAME}_index.csv"
)
OUTPUT_SIMILARITY_PATH = (
    "../outputs/embeddings_data/"
    f"day17_prompt_similarity_{MODEL_FILE_NAME}.csv"
)

TEXT_COLUMN = "text"
GROUP_COLUMN = "group_id"
PROMPT_TYPE_COLUMN = "prompt_type"
REQUIRED_PROMPT_TYPES = ["Question", "Instruction", "Injection"]


# =========================
# 2. 读取并检查输入数据
# =========================
print(f"正在读取 CSV 文件: {INPUT_CSV_PATH} ...")

try:
    # 优先尝试 UTF-8；失败后再尝试 gb18030，兼容不同编码的 CSV 文件。
    try:
        df = pd.read_csv(INPUT_CSV_PATH, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_CSV_PATH, encoding="gb18030")
except Exception as exc:
    print(f"读取 CSV 文件失败: {exc}")
    sys.exit(1)

# 检查后续处理必须使用的字段是否存在。
required_columns = {GROUP_COLUMN, PROMPT_TYPE_COLUMN, TEXT_COLUMN}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    print(f"CSV 缺少必要字段: {sorted(missing_columns)}")
    sys.exit(1)

# 文本不能是空值，否则无法生成有效的 Embedding。
if df[TEXT_COLUMN].isna().any():
    empty_rows = df.index[df[TEXT_COLUMN].isna()].tolist()
    print(f"字段 {TEXT_COLUMN!r} 存在空值，所在行索引: {empty_rows}")
    sys.exit(1)

# 转成字符串，避免数字等非字符串值传给 Embedding 模型。
df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)

# 去除 prompt_type 可能存在的首尾空格，保证类型匹配准确。
df[PROMPT_TYPE_COLUMN] = df[PROMPT_TYPE_COLUMN].astype(str).str.strip()

# 检查是否出现题目规定之外的 Prompt 类型。
unknown_prompt_types = sorted(
    set(df[PROMPT_TYPE_COLUMN].unique()) - set(REQUIRED_PROMPT_TYPES)
)
if unknown_prompt_types:
    print(f"发现未知 prompt_type: {unknown_prompt_types}")
    sys.exit(1)

# 检查每个 group_id 是否恰好包含 Question、Instruction、Injection 各一条。
invalid_groups = []
for group_id, group_df in df.groupby(GROUP_COLUMN, sort=False):
    type_counts = group_df[PROMPT_TYPE_COLUMN].value_counts().to_dict()
    is_valid = all(type_counts.get(prompt_type, 0) == 1 for prompt_type in REQUIRED_PROMPT_TYPES)
    if len(group_df) != 3 or not is_valid:
        invalid_groups.append(
            {
                "group_id": group_id,
                "row_count": len(group_df),
                "prompt_type_counts": type_counts,
            }
        )

if invalid_groups:
    print("以下分组不符合每组 Question、Instruction、Injection 各一条的要求：")
    for item in invalid_groups:
        print(item)
    sys.exit(1)

print(
    f"成功读取 {len(df)} 条数据，共 {df[GROUP_COLUMN].nunique()} 组；"
    "每组均包含 Question、Instruction、Injection 各一条。"
)


# =========================
# 3. 为每一条 text 单独生成 Embedding
# =========================
print(f"正在加载 Embedding 模型: {MODEL_NAME} ...")
# 首次运行时，SentenceTransformer 会自动下载模型。
model = SentenceTransformer(MODEL_NAME)

texts = df[TEXT_COLUMN].tolist()
print(f"开始为 {len(texts)} 条文本逐条生成 Embedding ...")

embeddings = model.encode(
    texts,
    batch_size=32,             # 可根据显存或内存大小调整。
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True, # 将向量归一化，点积即可等于余弦相似度。
)

# 保证向量为二维数组：[文本数量, Embedding 维度]。
embeddings = np.asarray(embeddings, dtype=np.float32)
if embeddings.ndim != 2 or embeddings.shape[0] != len(df):
    print(
        "Embedding 结果形状异常："
        f"得到 {embeddings.shape}，预期第一维为 {len(df)}。"
    )
    sys.exit(1)

print(f"Embedding 生成完成，向量形状: {embeddings.shape}")


# =========================
# 4. 保存 Embedding 及其行号映射
# =========================
# 自动创建输出目录，避免目录不存在时保存失败。
for output_path in [
    OUTPUT_EMBEDDING_PATH,
    OUTPUT_EMBEDDING_INDEX_PATH,
    OUTPUT_SIMILARITY_PATH,
]:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

# .npy 中第 i 行向量，严格对应 df 中第 i 行数据。
np.save(OUTPUT_EMBEDDING_PATH, embeddings)
print(f"Embedding 已保存到: {OUTPUT_EMBEDDING_PATH}")

# 单独保存索引映射，便于以后加载 .npy 后确认每个向量对应哪条原始数据。
embedding_index_df = df.copy()
embedding_index_df.insert(0, "embedding_index", np.arange(len(df)))
embedding_index_df.to_csv(
    OUTPUT_EMBEDDING_INDEX_PATH,
    index=False,
    encoding="utf-8-sig",
)
print(f"Embedding 行号映射已保存到: {OUTPUT_EMBEDDING_INDEX_PATH}")


# =========================
# 5. 计算每组三条 Prompt 的余弦相似度
# =========================
# 因为生成 Embedding 时使用了 normalize_embeddings=True，
# 两个单位向量的点积就是它们的余弦相似度。
# 为增强代码鲁棒性，下面仍显式处理分母，避免极端情况下出现零向量。
def cosine_similarity(vector_a, vector_b):
    """计算两个一维向量的余弦相似度。"""
    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denominator == 0:
        return np.nan
    return float(np.dot(vector_a, vector_b) / denominator)


similarity_rows = []

# groupby(sort=True) 让输出按 group_id 排序，方便检查结果。
for group_id, group_df in df.groupby(GROUP_COLUMN, sort=True):
    # 原始 DataFrame 的行索引与 embeddings 的行号完全一致。
    question_index = group_df.index[
        group_df[PROMPT_TYPE_COLUMN] == "Question"
    ][0]
    instruction_index = group_df.index[
        group_df[PROMPT_TYPE_COLUMN] == "Instruction"
    ][0]
    injection_index = group_df.index[
        group_df[PROMPT_TYPE_COLUMN] == "Injection"
    ][0]

    question_embedding = embeddings[question_index]
    instruction_embedding = embeddings[instruction_index]
    injection_embedding = embeddings[injection_index]

    similarity_rows.append(
        {
            GROUP_COLUMN: group_id,
            "question_text": df.at[question_index, TEXT_COLUMN],
            "instruction_text": df.at[instruction_index, TEXT_COLUMN],
            "injection_text": df.at[injection_index, TEXT_COLUMN],
            "similarity_question_instruction": cosine_similarity(
                question_embedding,
                instruction_embedding,
            ),
            "similarity_instruction_injection": cosine_similarity(
                instruction_embedding,
                injection_embedding,
            ),
            "similarity_question_injection": cosine_similarity(
                question_embedding,
                injection_embedding,
            ),
        }
    )

similarity_df = pd.DataFrame(similarity_rows)

# 保存每组的三项相似度。最终应为每个 group_id 一行。
similarity_df.to_csv(
    OUTPUT_SIMILARITY_PATH,
    index=False,
    encoding="utf-8-sig",
    float_format="%.6f",
)

print(f"余弦相似度结果已保存到: {OUTPUT_SIMILARITY_PATH}")
print("\n相似度结果预览：")
print(similarity_df.head().to_string(index=False))
print("\n✅ 全部处理完成。")


'''
day17_embeddings_bgeM3.npy
Embedding 向量文件。里面保存的是 CSV 中每一条 text 对应的向量，顺序与原始 CSV 行顺序一致。

day17_embeddings_bgeM3_index.csv
这是Embedding 与原始数据的对应关系表。
因为 .npy 文件只保存数字向量，本身看不出来某一行向量属于哪条文本，所以需要这个索引文件辅助解释.注意embedding_index列，.npy 和 _index.csv 是配套使用的。

day17_prompt_similarity_bgeM3.csv每组三类 Prompt 相似度结果文件。每一行代表一组数据，包含三种两两比较结果：
越接近 1：语义越相似
越接近 0：语义关系较弱
小于 0：语义方向可能相反


即：
.npy= 真正的向量数据

_index.csv= 向量对应哪条原始文本

_similarity.csv= 每组三条文本两两比较后的最终实验结果






'''