"""
数据预处理脚本
"""

import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

def load_parquet_safe(file_path):
    """安全加载 parquet 文件"""
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在。")
        return None
    try:
        df = pd.read_parquet(file_path)
        return df
    except Exception as e:
        print(f"读取 {file_path} 时出错: {e}")
        return None



def analyze_text_length(df, text_col='text', name="Dataset"):
    """专门分析文本列的长度分布并绘图"""
    if text_col not in df.columns:
        print(f"⚠️ 未找到字段 '{text_col}'，跳过文本长度分析。")
        return

    # 确保是字符串类型，处理可能的 NaN
    series = df[text_col].dropna().astype(str)

    if series.empty:
        print(f"⚠️ 字段 '{text_col}' 为空，跳过分析。")
        return

    # 计算字符长度
    lengths = series.str.len()

    # 1. 打印统计信息
    print(f"{name} 数据集 文本各类长度统计")
    print(f"\n📝 文本长度统计 (字段: '{text_col}'):")
    print(f"   - 最短长度: {lengths.min()}")
    print(f"   - 最长长度: {lengths.max()}")
    print(f"   - 平均长度: {lengths.mean():.2f}")
    print(f"   - 中位长度: {lengths.median():.2f}")
    print(f"   - 标准差:   {lengths.std():.2f}")

    # 打印一些分位数，帮助理解长尾分布
    print(f"   - 90% 数据小于: {lengths.quantile(0.9):.0f} 字符")
    print(f"   - 95% 数据小于: {lengths.quantile(0.95):.0f} 字符")
    print(f"   - 99% 数据小于: {lengths.quantile(0.99):.0f} 字符")

    # 2. 绘制直方图
    try:
        plt.figure(figsize=(10, 6))
        # 使用 hist 绘制，bins 可以根据数据量动态调整，这里设为 50 个区间
        plt.hist(lengths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)

        # 添加平均线和 median 线
        mean_val = lengths.mean()
        median_val = lengths.median()
        plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.0f}')
        plt.axvline(median_val, color='green', linestyle='-', label=f'Median: {median_val:.0f}')

        plt.title(f'Distribution of Text Lengths in "{text_col}" in {name}')
        plt.xlabel('Character Count')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)

        # 保存图表到本地，方便查看
        output_name = f"text_length_hist_{text_col}_{name}.png"
        file_dir = f"../outputs/text_length/"+output_name
        plt.savefig(file_dir, dpi=100, bbox_inches='tight')
        print(f"   📊 直方图已保存为: {file_dir}")

        # 如果在 Jupyter 或支持 UI 的环境，可以直接显示
        # plt.show()

        plt.close() # 关闭图形以释放内存
    except Exception as e:
        print(f"   ⚠️ 绘图失败: {e}")




def analyze_dataset(df, name="Dataset"):
    """对单个数据集进行详细分析"""
    if df is None or df.empty:
        print(f"{name} 为空或加载失败。\n")
        return

    print("="*30)
    print(f"📊 数据集分析: {name}")
    print("="*30)

    # 1. 基本形状
    rows, cols = df.shape
    print(f"📏 样本数量 (Rows): {rows}")
    print(f"📏 字段数量 (Cols): {cols}")

    # 2. 字段信息
    print("\n📋 字段列表及类型:")
    for col in df.columns:
        print(f"   - {col}: {df[col].dtype}")

    # 3. 数据预览 (真实长相)
    print("\n👀 数据预览 (前 5 条):")
    # 设置显示选项以确保能看到完整内容
    with pd.option_context('display.max_columns', None, 'display.width', 1000, 'display.max_colwidth', 50):
        print(df.head(5))

    # 4. 缺失值统计
    missing = df.isnull().sum()
    missing_total = missing.sum()
    if missing_total > 0:
        print(f"\n⚠️ 缺失值统计 (总缺失: {missing_total}):")
        print(missing[missing > 0])
    else:
        print("\n✅ 无缺失值")

    # 5. 标签分布分析
    # 策略：尝试寻找名为 'label', 'target', 'class' 的列，或者默认取最后一列作为标签
    label_col = None
    candidate_cols = ['label', 'target', 'class', 'y']

    for c in candidate_cols:
        if c in df.columns:
            label_col = c
            break

    if label_col is None:
        # 如果没有标准命名，假设最后一列是标签
        label_col = df.columns[-1]
        print(f"\n💡 未找到标准标签列名，默认假设最后一列 '{label_col}' 为标签。")
    else:
        print(f"\n🏷️ 检测到标签列: '{label_col}'")

    if label_col:
        print(f"\n📈 标签分布 ({label_col}):")
        value_counts = df[label_col].value_counts()
        # 打印分布
        print(value_counts)

        # 如果是数值型标签，打印一些统计描述
        if pd.api.types.is_numeric_dtype(df[label_col]):
            print(f"\n📉 标签数值统计:")
            print(df[label_col].describe())

    print("\n" + "="*30 + "\n")


    attack = df[df["label"]==1]
    print(f"Attack Sample By {name}:\n{attack.sample(10)}")
    safe = df[df["label"]==0]
    print(f"Safe Sample By {name}:\n{safe.sample(10)}")


    # 6. 【新增】文本长度分析
    # 检查是否存在 'text' 字段，或者第一个字符串类型的字段
    text_col_to_analyze = 'text'
    if text_col_to_analyze not in df.columns:
        # 如果没找到 'text'，尝试找其他看起来像文本的列
        string_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        if string_cols:
            text_col_to_analyze = string_cols[0]
            print(f"\n💡 未找到 'text' 字段，自动选择第一个文本字段: '{text_col_to_analyze}'")
        else:
            print("\n⚠️ 未找到任何文本字段，跳过文本长度分析。")
            text_col_to_analyze = None

    if text_col_to_analyze:
        analyze_text_length(df, text_col_to_analyze, name)



    print("\n" + "="*30 + "\n")




def analyze_dataset_test():
    # 1. 读取 train.parquet 文件
    # 如果只想读取 text 和 label 两个字段以节省内存，可以指定 columns 参数
    df = pd.read_parquet("../datasets/promptInjection/test.parquet", columns=["text", "label"])

    # 2. 将数据写入 CSV 文件
    # index=False 表示不将行索引写入 CSV 文件中
    df.to_csv("test.csv", index=False, encoding="utf-8-sig")

    print("转换完成！已生成 test.csv 文件。")




def main():

    train_file = "../datasets/promptInjection/train.parquet"
    test_file = "../datasets/promptInjection/test.parquet"

    print("🚀 开始自动化数据统计...\n")

    # 读取训练集
    df_train = load_parquet_safe(train_file)
    analyze_dataset(df_train, "Train Set")

    # 读取测试集
    df_test = load_parquet_safe(test_file)
    analyze_dataset(df_test, "Test Set")





if __name__ == "__main__":
    # main()
    analyze_dataset_test()
