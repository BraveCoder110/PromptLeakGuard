"""
Data preprocessing script
"""

import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

def load_parquet_safe(file_path):
    """loading parquet"""
    if not os.path.exists(file_path):
        print(f"error: file {file_path} is not exist。")
        return None
    try:
        df = pd.read_parquet(file_path)
        return df
    except Exception as e:
        print(f"reading {file_path} error: {e}")
        return None



def analyze_text_length(df, text_col='text', name="Dataset"):
    """Specifically analyze and plot the length distribution of text columns."""
    if text_col not in df.columns:
        print(f"not found '{text_col}'，Skip text length analysis。")
        return

    # Ensure the value is a string and handle potential NaN values.
    series = df[text_col].dropna().astype(str)

    if series.empty:
        print(f" '{text_col}' is empty，skip。")
        return

    # Calculate character length
    lengths = series.str.len()

    # 1. info
    print(f"{name} Dataset: Statistics on Text Lengths by Category")
    print(f"\n Text Length Statistics (field: '{text_col}'):")
    print(f"- min lengths: {lengths.min()}")
    print(f"- max lengths: {lengths.max()}")
    print(f"- mean len: {lengths.mean():.2f}")
    print(f"- median len: {lengths.median():.2f}")
    print(f"- std:   {lengths.std():.2f}")

    # Print some quantiles to help understand long-tailed distributions.
    print(f"- 90% Data is less than: {lengths.quantile(0.9):.0f} character")
    print(f"- 95% Data is less than: {lengths.quantile(0.95):.0f} character")
    print(f"- 99% Data is less than: {lengths.quantile(0.99):.0f} character")

    # 2. Plot a histogram
    try:
        plt.figure(figsize=(10, 6))
        # Plotted using `hist`, where the number of bins can be dynamically adjusted based on the data volume; here, it is set to 50 bins.
        plt.hist(lengths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)

        # Add mean and median lines.
        mean_val = lengths.mean()
        median_val = lengths.median()
        plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.0f}')
        plt.axvline(median_val, color='green', linestyle='-', label=f'Median: {median_val:.0f}')

        plt.title(f'Distribution of Text Lengths in "{text_col}" in {name}')
        plt.xlabel('Character Count')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)

        # Save the chart locally for easy viewing.
        output_name = f"text_length_hist_{text_col}_{name}.png"
        file_dir = f"../outputs/text_length/"+output_name
        plt.savefig(file_dir, dpi=100, bbox_inches='tight')
        print(f" saved to: {file_dir}")

        # plt.show()

        plt.close()
    except Exception as e:
        print(f"failed: {e}")




def analyze_dataset(df, name="Dataset"):
    """Conduct a detailed analysis of a single dataset."""
    if df is None or df.empty:
        print(f"{name} empty or failed to load。\n")
        return

    print("="*30)
    print(f" Dataset Analysis: {name}")
    print("="*30)

    # 1. shape
    rows, cols = df.shape
    print(f" Rows: {rows}")
    print(f" Cols: {cols}")

    # 2. fields info
    print("\n Field List and Types:")
    for col in df.columns:
        print(f"- {col}: {df[col].dtype}")

    # 3. Data Preview (Actual Appearance)
    print("\n Data Preview (First 5 Rows):")
    # Adjust display settings to ensure the full content is visible.
    with pd.option_context('display.max_columns', None, 'display.width', 1000, 'display.max_colwidth', 50):
        print(df.head(5))

    # 4. Missing Value Statistics
    missing = df.isnull().sum()
    missing_total = missing.sum()
    if missing_total > 0:
        print(f"\n Missing Value Statistics (Total loss: {missing_total}):")
        print(missing[missing > 0])
    else:
        print("\n no missing values")

    # 5. Label distribution analysis
    # Strategy: Attempt to locate columns named 'label', 'target', or 'class', or default to using the last column as the label.
    label_col = None
    candidate_cols = ['label', 'target', 'class', 'y']

    for c in candidate_cols:
        if c in df.columns:
            label_col = c
            break

    if label_col is None:
        # In the absence of standard naming, assume the last column is the label.
        label_col = df.columns[-1]
        print(f"\n 未找到标准标签列名，默认假设最后一列 '{label_col}' 为标签。")
    else:
        print(f"\n️ 检测到标签列: '{label_col}'")

    if label_col:
        print(f"\n 标签分布 ({label_col}):")
        value_counts = df[label_col].value_counts()
        #
        print(value_counts)

        # If the label is numerical, print some descriptive statistics.
        if pd.api.types.is_numeric_dtype(df[label_col]):
            print(f"\n Tag Value Statistics:")
            print(df[label_col].describe())

    print("\n" + "="*30 + "\n")


    attack = df[df["label"]==1]
    print(f"Attack Sample By {name}:\n{attack.sample(10)}")
    safe = df[df["label"]==0]
    print(f"Safe Sample By {name}:\n{safe.sample(10)}")


    # 6. [New] Text Length Analysis
    # Check for the existence of a 'text' field, or the first field of string type.
    text_col_to_analyze = 'text'
    if text_col_to_analyze not in df.columns:
        # If 'text' is not found, try looking for other columns that appear to contain text.
        string_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        if string_cols:
            text_col_to_analyze = string_cols[0]
            print(f"\n 'text' field not found; automatically selecting the first text field.: '{text_col_to_analyze}'")
        else:
            print("\n No text fields found; skipping text length analysis。")
            text_col_to_analyze = None

    if text_col_to_analyze:
        analyze_text_length(df, text_col_to_analyze, name)

    print("\n" + "="*30 + "\n")




def analyze_dataset_test():
    # 1. read train.parquet
    # If you only want to read the `text` and `label` fields to save memory, you can specify the `columns` parameter.
    df = pd.read_parquet("../datasets/promptInjection/test.parquet", columns=["text", "label"])

    # 2.Write data to a CSV file.
    # index=False Indicates that the row index should not be written to the CSV file.
    df.to_csv("test.csv", index=False, encoding="utf-8-sig")

    print("转换完成！已生成 test.csv 文件。")




def main():

    train_file = "../datasets/promptInjection/train.parquet"
    test_file = "../datasets/promptInjection/test.parquet"

    print(" Start automated data statistics...\n")

    # Load the training set.
    df_train = load_parquet_safe(train_file)
    analyze_dataset(df_train, "Train Set")

    # Load the test set
    df_test = load_parquet_safe(test_file)
    analyze_dataset(df_test, "Test Set")


if __name__ == "__main__":
    # main()
    analyze_dataset_test()
