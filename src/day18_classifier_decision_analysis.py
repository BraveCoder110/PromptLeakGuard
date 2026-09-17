"""
Day 18 Task 4: Decision Analysis of Logistic Regression Classifier.

Functions:
1. Load previously saved BGE-M3 embeddings;
2. Train a binary Logistic Regression classifier (Safe vs. Attack) using `expected_label`;
3. Generate instance-level OOF (Out-of-Fold) decision scores using GroupKFold;
4. Compare average decision scores across three prompt categories: Question, Instruction, and Injection;
5. Save instance-level results, aggregated category results, and the final classifier model.

Decision Score Interpretation:
- "Attack" is encoded as the positive class (1), and "Safe" as the negative class (0);
- A higher decision score indicates a stronger model tendency to classify the input as Attack/Injection;
- A decision score below 0 indicates a tendency to classify the input as Safe;
- The decision score is not a probability, but it can be converted via the sigmoid function into an uncalibrated "Attack propensity" value.
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
# 1. Path and Field Configuration
# =========================
# When the script is placed in the project code directory, BASE_DIR can be adjusted according to the project structure.
SCRIPT_DIR = Path(__file__).resolve().parent

# Prioritizes the current upload directory; also maintains compatibility with the `../outputs` path from the original project.
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
    """Find the first file that actually exists among the candidate paths."""
    for path in candidates:
        if path.exists():
            return path

    print(f"not found {description}：")
    for path in candidates:
        print(f"  - {path}")
    sys.exit(1)


# =========================
# 2. Load Embeddings and Index Table
# =========================
embedding_path = find_existing_path(
    CANDIDATE_EMBEDDING_PATHS,
    "Embedding file",
)
index_path = find_existing_path(
    CANDIDATE_INDEX_PATHS,
    "Embedding index CSV",
)

print(f"reading Embedding：{embedding_path}")
embeddings = np.load(embedding_path)
embeddings = np.asarray(embeddings, dtype=np.float32)

print(f"reading index data：{index_path}")
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
    print(f"Index CSV Missing required fields：{sorted(missing_columns)}")
    sys.exit(1)

if embeddings.ndim != 2:
    print(f"Embedding Must be a two-dimensional array，shape：{embeddings.shape}")
    sys.exit(1)

if len(index_df) != embeddings.shape[0]:
    print(
        "Mismatch between the number of embeddings and the number of rows in the index CSV:"
        f"Embedding={embeddings.shape[0]}，CSV={len(index_df)}"
    )
    sys.exit(1)

# If the index file contains an embedding_index, check if it is consistent with the .npy row order.
if "embedding_index" in index_df.columns:
    expected_indexes = np.arange(len(index_df))
    actual_indexes = index_df["embedding_index"].to_numpy()
    if not np.array_equal(actual_indexes, expected_indexes):
        print("embedding_index Since the row numbers are not consecutive and do not start from 0, correct vector mapping cannot be guaranteed.")
        sys.exit(1)

# Clean string fields to prevent category matching failures caused by whitespace.
index_df[PROMPT_TYPE_COLUMN] = (
    index_df[PROMPT_TYPE_COLUMN].astype(str).str.strip()
)
index_df[LABEL_COLUMN] = index_df[LABEL_COLUMN].astype(str).str.strip()

unknown_prompt_types = sorted(
    set(index_df[PROMPT_TYPE_COLUMN].unique()) - set(REQUIRED_PROMPT_TYPES)
)
if unknown_prompt_types:
    print(f"unknown prompt_type：{unknown_prompt_types}")
    sys.exit(1)

unknown_labels = sorted(
    set(index_df[LABEL_COLUMN].unique()) - {NEGATIVE_LABEL, POSITIVE_LABEL}
)
if unknown_labels:
    print(f"unknown expected_label：{unknown_labels}")
    sys.exit(1)


# =========================
# 3. Construct binary classification target
# =========================
# Safe -> 0, Attack -> 1.
# Since Attack is the positive class, a higher decision_function score indicates a stronger tendency towards Attack.
y = (index_df[LABEL_COLUMN] == POSITIVE_LABEL).astype(int).to_numpy()
groups = index_df[GROUP_COLUMN].to_numpy()

if len(np.unique(y)) != 2:
    print("The data must include both 'Safe' and 'Attack' categories.")
    sys.exit(1)

number_of_groups = index_df[GROUP_COLUMN].nunique()
n_splits = min(5, number_of_groups)
if n_splits < 2:
    print("At least two group_ids are required to perform GroupKFold.")
    sys.exit(1)


# =========================
# 4. Define the Logistic Regression classifier
# =========================
# Embedding dimensions typically have similar scales, but standardization helps stabilize training for linear models.
# Setting with_mean=False avoids centering the matrix, ensuring compatibility with sparse matrices that might be used in the future.
classifier = make_pipeline(
    StandardScaler(with_mean=False),
    LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
)


# =========================
# 5. Generate OOF decision scores
# =========================
# We do not directly use classifier.fit(...).decision_function(embeddings) as the reported result,
# because those are in-sample scores and may be overly optimistic.
#
# GroupKFold ensures that the three prompts sharing the same group_id always fall into the same fold:
# the Question, Instruction, and Injection components are not split between the training and validation sets,
# thereby reducing data leakage caused by high semantic similarity within the same group.
group_cv = GroupKFold(n_splits=n_splits)
print(f"Calculating OOF decision scores using {n_splits}-fold GroupKFold...")

oof_decision_scores = cross_val_predict(
    classifier,
    embeddings,
    y,
    groups=groups,
    cv=group_cv,
    method="decision_function",
)
oof_decision_scores = np.asarray(oof_decision_scores, dtype=float).reshape(-1)

# Convert the decision score into a propensity value between 0 and 1 using the sigmoid function.
# This value has not undergone probability calibration; therefore, it is termed "attack tendency" rather than a formal probability.
oof_attack_tendency = 1.0 / (1.0 + np.exp(-oof_decision_scores))


# =========================
# 6. Save the results of each individual decision.
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
# 7. Grouped by prompt type
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

# The output order is fixed to facilitate direct observation of the changes in the Question -> Instruction -> Injection sequence.
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
# 8. Save the file and the final model.
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

# Train the final model using the full dataset for future embedding generation on new text:
# classifier.decision_function(new_embeddings)
classifier.fit(embeddings, y)
joblib.dump(classifier, OUTPUT_MODEL_PATH)

print("\nSummary of classifier decisions for the three categories of prompts:")
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

print("\nresult：")
print("The higher the mean_decision_score, the more it leans towards Attack/Injection.")
print("Typically, the expectation is that 'Question' scores are lower, 'Instruction' scores are in the middle, and 'Injection' scores are higher.")
print("the final ranking must be based on the actual calculation results and cannot be assumed in advance.")

print("\n saved as：")
print(f"Individual results：{OUTPUT_DETAIL_PATH}")
print(f"Subtotals：{OUTPUT_SUMMARY_PATH}")
print(f"Final model：{OUTPUT_MODEL_PATH}")
