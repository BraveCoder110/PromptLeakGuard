"""
Generate embeddings and complete Day 18 Task 3: Embedding space inspection.

Input data is from Day 17: `../docs/day17_instruction_bias.csv`
Each `group_id` should contain three prompts: Question, Instruction, and Injection.

The program will output:
1. Embeddings corresponding to each text entry: `.npy` file
2. Mapping between embeddings and original data: CSV file
3. Pairwise cosine similarities between the three types of prompts within each group: CSV file
"""

import os
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# =========================
# 1. Configuration Parameters
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
# 2. Read and check the input data.
# =========================
print(f"reading CSV file: {INPUT_CSV_PATH} ...")

try:
    # Try UTF-8 first; if that fails, try gb18030 to ensure compatibility with CSV files using different encodings.
    try:
        df = pd.read_csv(INPUT_CSV_PATH, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_CSV_PATH, encoding="gb18030")
except Exception as exc:
    print(f"failed: {exc}")
    sys.exit(1)

# Check for the existence of fields required for subsequent processing.
required_columns = {GROUP_COLUMN, PROMPT_TYPE_COLUMN, TEXT_COLUMN}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    print(f"CSV Missing required fields: {sorted(missing_columns)}")
    sys.exit(1)

# The text cannot be null; otherwise, a valid embedding cannot be generated.
if df[TEXT_COLUMN].isna().any():
    empty_rows = df.index[df[TEXT_COLUMN].isna()].tolist()
    print(f"field {TEXT_COLUMN!r} Contains null values,Index of the row: {empty_rows}")
    sys.exit(1)

# Convert to a string to avoid passing non-string values, such as numbers, to the embedding model.
df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)

# Remove any leading or trailing whitespace from `prompt_type` to ensure an accurate type match.
df[PROMPT_TYPE_COLUMN] = df[PROMPT_TYPE_COLUMN].astype(str).str.strip()

# Check for the presence of prompt types other than those specified in the problem requirements.
unknown_prompt_types = sorted(
    set(df[PROMPT_TYPE_COLUMN].unique()) - set(REQUIRED_PROMPT_TYPES)
)
if unknown_prompt_types:
    print(f"unknown prompt_type: {unknown_prompt_types}")
    sys.exit(1)

# Check whether each `group_id` contains exactly one instance each of Question, Instruction, and Injection.
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
    print("The following groupings do not meet the requirement of having one Question, one Instruction, and one Injection per group.：")
    for item in invalid_groups:
        print(item)
    sys.exit(1)

print(
    f"Successfully read {len(df)} ，total： {df[GROUP_COLUMN].nunique()} ；"
    "Each set contains one Question, one Instruction, and one Injection."
)


# =========================
# 3. Generate an embedding for each text string individually.
# =========================
print(f"Loading embedding model...: {MODEL_NAME} ...")
# SentenceTransformer automatically downloads the model upon the first run.
model = SentenceTransformer(MODEL_NAME)

texts = df[TEXT_COLUMN].tolist()
print(f"len: {len(texts)} , Embedding ...")

embeddings = model.encode(
    texts,
    batch_size=32,             # It can be adjusted based on the size of the video memory or system memory.
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True, # By normalizing the vectors, the dot product becomes equal to the cosine similarity.
)

# Ensure the vectors are a 2D array: [number of texts, embedding dimension].
embeddings = np.asarray(embeddings, dtype=np.float32)
if embeddings.ndim != 2 or embeddings.shape[0] != len(df):
    print(
        "Embedding result shape is abnormal.："
        f"shape: {embeddings.shape}，The expected first dimension is {len(df)}。"
    )
    sys.exit(1)

print(f"Embedding Generation complete, shape: {embeddings.shape}")


# =========================
# 4. Save embeddings and their row number mappings
# =========================
# Automatically create the output directory to prevent save failures if the directory does not exist.
for output_path in [
    OUTPUT_EMBEDDING_PATH,
    OUTPUT_EMBEDDING_INDEX_PATH,
    OUTPUT_SIMILARITY_PATH,
]:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

np.save(OUTPUT_EMBEDDING_PATH, embeddings)
print(f"Embedding saved as: {OUTPUT_EMBEDDING_PATH}")

# Save the index mapping separately to facilitate identifying which original data point corresponds to each vector after loading the .npy file.
embedding_index_df = df.copy()
embedding_index_df.insert(0, "embedding_index", np.arange(len(df)))
embedding_index_df.to_csv(
    OUTPUT_EMBEDDING_INDEX_PATH,
    index=False,
    encoding="utf-8-sig",
)
print(f"Embedding saved as: {OUTPUT_EMBEDDING_INDEX_PATH}")


# =========================
# 5. Calculate the cosine similarity for each group of three prompts
# =========================
# Since `normalize_embeddings=True` was used during embedding generation,
# the dot product of two unit vectors is equivalent to their cosine similarity.
# To enhance code robustness, the denominator is still explicitly handled below
# to avoid issues with zero vectors in extreme cases.
def cosine_similarity(vector_a, vector_b):
    """Calculate the cosine similarity between two one-dimensional vectors.。"""
    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denominator == 0:
        return np.nan
    return float(np.dot(vector_a, vector_b) / denominator)


similarity_rows = []

# `groupby(sort=True)` sorts the output by `group_id`, making it easier to inspect the results.
for group_id, group_df in df.groupby(GROUP_COLUMN, sort=True):
    # The row index of the original DataFrame corresponds exactly to the row numbers of the embeddings.
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

# Save the three similarity scores for each group. The final result should have one row per `group_id`.
similarity_df.to_csv(
    OUTPUT_SIMILARITY_PATH,
    index=False,
    encoding="utf-8-sig",
    float_format="%.6f",
)

print(f"Cosine similarity results have been saved to: {OUTPUT_SIMILARITY_PATH}")
print("\nSimilarity Result Preview：")
print(similarity_df.head().to_string(index=False))
print("\nAll processing completed.。")


'''
day17_embeddings_bgeM3.npy
Embedding vector file. It stores the vectors corresponding to each text entry in the CSV, maintaining the same order as the rows in the original CSV.

day17_embeddings_bgeM3_index.csv
This is the mapping table linking embeddings to the original data.
Since the .npy file stores only numerical vectors—making it impossible to tell which text a specific vector belongs to just by looking at it—this index file is required for interpretation. Pay attention to the `embedding_index` column; the .npy file and the `_index.csv` file are designed to be used together.

day17_prompt_similarity_bgeM3.csv
File containing prompt similarity results for groups of three. Each row represents a data group and includes the results of pairwise comparisons between the three items:
Closer to 1: Semantically more similar
Closer to 0: Weaker semantic relationship
Less than 0: Semantic directions may be opposite


In summary:
.npy = The actual vector data

_index.csv = Mapping of vectors to their corresponding original text

_similarity.csv = Final experimental results from pairwise comparisons of the three texts in each group
'''