'''
Logistic Regression Analysis for Prompt Injection Detection
基于train_lr_baseline.py导出 FN (假阴性) 和 FP (假阳性) 样本 并分析
Export and analyze FN (false negative) and FP (false positive) samples based on train_lr_baseline.py.
'''

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def export_hard_cases(train_npy, test_npy, train_parquet, test_parquet,
                      label_col="label", text_col="text", model_name="lr_baseline"):
    """
    Train the model and export FN (false negative) and FP (false positive) samples.
    """
    # 1. loading
    print("loading data...")
    X_train = np.load(train_npy)
    X_test = np.load(test_npy)

    df_train = pd.read_parquet(train_parquet)
    df_test = pd.read_parquet(test_parquet)

    y_train = df_train[label_col].values
    y_test = df_test[label_col].values

    # Ensure the text column exists.
    if text_col not in df_test.columns:
        raise ValueError(f"not found '{text_col}'，check Parquet 。")

    # 2. train
    print("train Logistic Regression...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    # 3. predict
    y_pred = clf.predict(X_test)

    # 4. Filter FNs and FPs
    # FN (False Negative): Actual is 1 (Attack), predicted is 0 (Safe) -> Missed detection
    fn_mask = (y_test == 1) & (y_pred == 0)
    fn_indices = np.where(fn_mask)[0]

    # FP (False Positive): Actual: 0 (Safe), Predicted: 1 (Attack) -> False Positive
    fp_mask = (y_test == 0) & (y_pred == 1)
    fp_indices = np.where(fp_mask)[0]

    print(f"\n the result:")
    print(f"FN (Missed detections/attacks treated as safe): {len(fn_indices)} ")
    print(f"FP (False positive / Security activity mistaken for an attack): {len(fp_indices)} ")

    # 5. Construct a detailed data table.
    def build_detail_df(indices, error_type):
        if len(indices) == 0:
            return pd.DataFrame()

        data = {
            "index": indices,
            "error_type": [error_type] * len(indices),
            "true_label": y_test[indices],
            "pred_label": y_pred[indices],
            "text": df_test.iloc[indices][text_col].values,
            # Optional: If you want to save the corresponding embeddings for subsequent analysis, you can uncomment the lines below.
            # "embedding": list(X_test[indices])
        }
        return pd.DataFrame(data)

    df_fn = build_detail_df(fn_indices, "FN (Missed Attack)")
    df_fp = build_detail_df(fp_indices, "FP (False Alarm)")

    # 6. export file
    output_prefix = f"{model_name}_hard_cases"

    if not df_fn.empty:
        csv_fn = f"{output_prefix}_FN.csv"
        df_fn.to_csv(csv_fn, index=False, encoding='utf-8-sig')
        print(f"FN saved to: {csv_fn}")
        print("FN sample  ---")
        print(df_fn[['text', 'true_label', 'pred_label']].head(3).to_string(index=False))

    if not df_fp.empty:
        csv_fp = f"{output_prefix}_FP.csv"
        df_fp.to_csv(csv_fp, index=False, encoding='utf-8-sig')
        print(f"FP saved to: {csv_fp}")
        print("FP sample ---")
        print(df_fp[['text', 'true_label', 'pred_label']].head(3).to_string(index=False))

    # You can also save a merged file.
    pd.concat([df_fn, df_fp]).to_csv(f"{output_prefix}_all.csv", index=False, encoding='utf-8-sig')
    print(f"All difficult cases have been merged and saved to: {output_prefix}_all.csv")




if __name__ == "__main__":
    export_hard_cases(
        train_npy="../outputs/embeddings_data/train_embeddings.npy",
        test_npy="../outputs/embeddings_data/test_embeddings.npy",
        train_parquet="../datasets/promptInjection/train.parquet",
        test_parquet="../datasets/promptInjection/test.parquet",
        text_col="text",      # The name of the text column in the Parquet file.
        label_col="label"     # The name of the label column in the Parquet file.
    )

'''
Analysis：

    攻击模式分布
    | 攻击类型 | 数量 | 占比 | 典型特征 |
    |---------|------|------|---------|
    | 提示注入 | 7条 | 36.8% | "Ignore the above", "Forget everything" |
    | SQL/技术请求 | 3条 | 15.8% | "Generate SQL code", "return embeddings" |
    | 角色扮演 | 2条 | 10.5% | "als Drehbuchautor agieren"(德语) |
    | 信息泄露 | 2条 | 10.5% | "when were you trained" |
    | 混合内容 | 2条 | 10.5% | 正常话题+攻击指令拼接 |
    | 短命令 | 2条 | 10.5% | "translate to polish" |
    | 其他 | 1条 | 5.3% | 直接德语攻击指令 |

Why do models fail to detect attacks? Six key reasons

1. Semantic dilution effect (most severe)
Attack instructions are embedded within a large amount of benign content; for example, in Sample 114 (345 characters), only the final sentence constitutes the attack.
Embeddings represent the semantic average of the entire text, causing the attack signal to be weakened to the point of being unrecognizable.

2. Cross-lingual challenges
26.3% of samples are entirely in German, and 10.5% are a mix of English and German.
The `all-MiniLM-L6-v2` model used was primarily trained on English and offers weak support for German.

3. Strong contextual dependency
Phrases like "Ignore what I said before" require an understanding of meta-instruction overrides.
Simple text similarity measures cannot capture attacks based on this type of logic.

4. Ambiguous "gray areas"
"Tell me a joke about a woman" – could simply be inappropriate humor.
"Translate to Polish" – benign in itself, but potentially used to bypass restrictions.

5. Underestimation of technical attacks
Technical requests such as SQL generation or embedding extraction appear neutral.
The model classifies them as "technical queries" rather than "policy-violating attacks."

6. Insufficient information in short texts
Very short commands (e.g., the 19-character "Translate to Polish") lack sufficient semantic features.
It is difficult for embeddings to extract malicious intent from such a limited vocabulary.

Prioritization of improvement suggestions

High priority (immediate implementation):
Rule-based fallback filtering: Forcefully flag samples containing keywords like "ignore," "forget," "instructions," or "embeddings" as attacks.
Data augmentation: Generate variants based on these seven prompt injection patterns to expand the training set.

Medium priority (short-term optimization):
Feature engineering: Add meta-features such as "contains injection keywords," "text length," and "language type."
Model replacement: Switch to a multilingual model like `paraphrase-multilingual-MiniLM` or a larger model like `all-mpnet-base-v2` (768 dimensions).

Low priority (long-term planning):
Hierarchical detection architecture: Keyword filtering → Embedding-based classification → Human review.
Ensemble learning: Combine LR, SVM, and Random Forest to improve robustness.

These false-negative (FN) samples reveal the inherent limitations of current embedding-based classifiers when handling covert attacks, cross-lingual scenarios, and context-dependent attacks.
'''