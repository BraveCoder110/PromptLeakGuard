'''
Logistic Regression Analysis for Prompt Injection Detection
基于train_lr_baseline.py导出 FN (假阴性) 和 FP (假阳性) 样本 并分析
'''

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def export_hard_cases(train_npy, test_npy, train_parquet, test_parquet,
                      label_col="label", text_col="text", model_name="lr_baseline"):
    """
    训练模型并导出 FN (假阴性) 和 FP (假阳性) 样本。
    """
    # 1. 加载数据
    print("🔄 正在加载数据...")
    X_train = np.load(train_npy)
    X_test = np.load(test_npy)

    df_train = pd.read_parquet(train_parquet)
    df_test = pd.read_parquet(test_parquet)

    y_train = df_train[label_col].values
    y_test = df_test[label_col].values

    # 确保文本列存在
    if text_col not in df_test.columns:
        raise ValueError(f"测试集中未找到文本列 '{text_col}'，请检查 Parquet 文件结构。")

    # 2. 训练模型
    print("🤖 正在训练 Logistic Regression...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    # 3. 预测
    y_pred = clf.predict(X_test)

    # 4. 筛选 FN 和 FP
    # FN (False Negative): 真实是 1 (Attack)，预测是 0 (Safe) -> 漏报
    fn_mask = (y_test == 1) & (y_pred == 0)
    fn_indices = np.where(fn_mask)[0]

    # FP (False Positive): 真实是 0 (Safe)，预测是 1 (Attack) -> 误报
    fp_mask = (y_test == 0) & (y_pred == 1)
    fp_indices = np.where(fp_mask)[0]

    print(f"\n📊 统计结果:")
    print(f"   - FN (漏报/攻击被当成安全): {len(fn_indices)} 条")
    print(f"   - FP (误报/安全被当成攻击): {len(fp_indices)} 条")

    # 5. 构建详细数据表
    def build_detail_df(indices, error_type):
        if len(indices) == 0:
            return pd.DataFrame()

        data = {
            "index": indices,
            "error_type": [error_type] * len(indices),
            "true_label": y_test[indices],
            "pred_label": y_pred[indices],
            "text": df_test.iloc[indices][text_col].values,
            # 可选：如果你想保存对应的 embedding 用于后续分析，可以取消下面注释
            # "embedding": list(X_test[indices])
        }
        return pd.DataFrame(data)

    df_fn = build_detail_df(fn_indices, "FN (Missed Attack)")
    df_fp = build_detail_df(fp_indices, "FP (False Alarm)")

    # 6. 导出文件
    output_prefix = f"{model_name}_hard_cases"

    if not df_fn.empty:
        csv_fn = f"{output_prefix}_FN.csv"
        df_fn.to_csv(csv_fn, index=False, encoding='utf-8-sig')
        print(f"✅ FN 样本已保存至: {csv_fn}")
        print("--- FN 样本预览 (前3条) ---")
        print(df_fn[['text', 'true_label', 'pred_label']].head(3).to_string(index=False))

    if not df_fp.empty:
        csv_fp = f"{output_prefix}_FP.csv"
        df_fp.to_csv(csv_fp, index=False, encoding='utf-8-sig')
        print(f"✅ FP 样本已保存至: {csv_fp}")
        print("--- FP 样本预览 (前3条) ---")
        print(df_fp[['text', 'true_label', 'pred_label']].head(3).to_string(index=False))

    # 也可以保存一个合并的文件
    pd.concat([df_fn, df_fp]).to_csv(f"{output_prefix}_all.csv", index=False, encoding='utf-8-sig')
    print(f"✅ 所有难例已合并保存至: {output_prefix}_all.csv")




if __name__ == "__main__":
    export_hard_cases(
        train_npy="../outputs/embeddings_data/train_embeddings.npy",
        test_npy="../outputs/embeddings_data/test_embeddings.npy",
        train_parquet="../datasets/promptInjection/train.parquet",
        test_parquet="../datasets/promptInjection/test.parquet",
        text_col="text",      # 你的 parquet 中文本列的名字
        label_col="label"     # 你的 parquet 中标签列的名字
    )


'''
导出后，你可以重点分析以下内容来优化模型：
分析 FN (19条)：
这些是最危险的样本。检查它们是否使用了隐晦的攻击方式、变体词或上下文依赖？
如果是，考虑在训练集中增加类似的负样本（Data Augmentation）。
分析 FP (1条)：
检查这条“安全”文本是否包含了敏感词但语境其实是安全的？
如果是，说明模型对关键词过于敏感，可能需要更多的上下文特征或更大的预训练模型。
'''



'''
分析：

    📊 攻击模式分布
    | 攻击类型 | 数量 | 占比 | 典型特征 |
    |---------|------|------|---------|
    | 提示注入 | 7条 | 36.8% | "Ignore the above", "Forget everything" |
    | SQL/技术请求 | 3条 | 15.8% | "Generate SQL code", "return embeddings" |
    | 角色扮演 | 2条 | 10.5% | "als Drehbuchautor agieren"(德语) |
    | 信息泄露 | 2条 | 10.5% | "when were you trained" |
    | 混合内容 | 2条 | 10.5% | 正常话题+攻击指令拼接 |
    | 短命令 | 2条 | 10.5% | "translate to polish" |
    | 其他 | 1条 | 5.3% | 直接德语攻击指令 |



🔍 为什么模型会漏报?六大核心原因

1. 语义稀释效应(最严重)
攻击指令被大量无害内容包裹,如样本114(345字符)只有最后一句是攻击
Embedding是整个文本的语义平均,攻击信号被弱化到无法识别

2. 跨语言挑战
26.3%的样本是纯德语,10.5%是英德混写
使用的 all-MiniLM-L6-v2 模型主要训练于英文,对德语支持弱

3. 上下文依赖性强
"Ignore what I said before" 需要理解这是元指令层面的覆盖操作
单纯的文本相似度无法捕捉这种逻辑层面的攻击

4. 边界模糊的灰色地带
"Tell me a joke about a woman" - 可能只是不当幽默
"translate to polish" - 本身无害但可能被用于绕过限制

5. 技术类攻击被低估
SQL生成、embedding提取等技术请求看似中性
模型将其归类为"技术问题"而非"违反政策的攻击"

6. 短文本信息不足
极短命令(如19字符的"translate to polish")缺乏足够语义特征
Embedding难以从少量词汇中提取攻击意图

💡 改进建议优先级

高优先级(立即实施):
规则过滤兜底:对包含 "ignore/forget/instructions/embeddings" 等关键词的样本强制标记为攻击
数据增强:针对这7种提示注入模式生成变体,扩充训练集

中优先级(短期优化):
特征工程:添加"是否含注入词"、"文本长度"、"语言类型"等元特征
更换模型:使用多语言模型 paraphrase-multilingual-MiniLM 或更大的 all-mpnet-base-v2(768维)

低优先级(长期规划):
分层检测架构:关键词过滤 → Embedding分类 → 人工审核
集成学习:结合 LR + SVM + Random Forest 提升鲁棒性

这些FN样本揭示了当前基于Embedding的分类器在处理隐蔽攻击、跨语言场景和上下文依赖型攻击时的固有局限性。

'''