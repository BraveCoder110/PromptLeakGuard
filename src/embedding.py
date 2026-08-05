"""
生成embedding

使用all-MiniLM-L6-v2模型作为默认模型，Baseline
"""
import sys
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def generate_and_save_embeddings(parquet_path, output_npy_path, text_column="text", model_name="all-MiniLM-L6-v2"):
    """
    读取 parquet 文件，生成文本 Embedding 并保存为 .npy 文件。
    """
    # 1. 加载 Embedding 模型
    # 首次运行会自动从 HuggingFace 下载模型
    print(f"正在加载模型: {model_name} ...")
    model = SentenceTransformer(model_name)

    # 2. 读取 Parquet 数据
    print(f"正在读取文件: {parquet_path} ...")
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"读取 Parquet 文件失败: {e}")
        return

    if text_column not in df.columns:
        print(f"错误: 找不到指定的文本列 '{text_column}'。当前列名为: {list(df.columns)}")
        return

    print(f"打印df看下数据:{df.head()}")

    # 提取文本列表，并处理可能存在的空值
    texts = df[text_column].astype(str).fillna("").tolist()
    print(f"成功读取 {len(texts)} 条数据，开始生成 Embedding...")

    # 3. 批量生成 Embedding 向量
    # show_progress_bar=True #会显示进度条，方便观察长耗时任务
    embeddings = model.encode(
        texts,
        batch_size=32,            # 可根据显存/内存大小调整
        show_progress_bar=True,
        convert_to_numpy=True     # 直接输出 numpy 数组格式
    )

    # 4. 保存为 .npy 文件
    print(f"正在保存 Embedding 到: {output_npy_path} ...")
    np.save(output_npy_path, embeddings)
    print(f"✅ 成功! 向量形状为: {embeddings.shape}")



def visualize_embeddings_2d(parquet_path, npy_path, file_type, label_column="label", text_column="text"):
    """
    读取 Parquet 中的标签和 npy 中的 Embedding，使用 PCA 降至 2 维并绘制散点图。
    """
    # 1. 加载数据
    print(f"正在加载文件: {parquet_path} 和 {npy_path} ...")
    try:
        df = pd.read_parquet(parquet_path)
        embeddings = np.load(npy_path)
    except Exception as e:
        print(f"文件加载失败: {e}")
        return

    # 2. 检查标签列是否存在
    if label_column not in df.columns:
        print(f"错误: 找不到标签列 '{label_column}'。当前列名为: {list(df.columns)}")
        return

    # 3. PCA 降维 (384维 -> 2维)
    print("正在进行 PCA 降维 (384D -> 2D)...")
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    print(f"✅ PCA 降维完成。解释方差比例: {pca.explained_variance_ratio_}")

    # 4. 绘制散点图
    plt.figure(figsize=(10, 8))

    # 获取所有唯一的标签
    unique_labels = df[label_column].unique()

    # print(unique_labels)
    # sys.exit(0)


    colors = {"Attack": "#e74c3c", "Safe": "#2ecc71"}  # 攻击用红色，安全用绿色

    for label in unique_labels:
        # 筛选出属于当前标签的 2D 坐标
        mask = (df[label_column] == label)
        x = embeddings_2d[mask, 0]
        y = embeddings_2d[mask, 1]

        # 兼容数字标签和字符串标签
        # 假设 1 或 'attack' 为攻击类，0 或 'safe' 为安全类
        if str(label).lower() in ["1", "attack"]:
            marker = '.'
            size = 50
            color = "#e74c3c"  # 攻击用红色
        elif str(label).lower() in ["0", "safe"]:
            marker = '^'
            size = 60
            color = "#2ecc71"  # 安全用绿色
        else:
            marker = 'o'  # 其他标签默认用圆点
            size = 50
            color = "#3498db"  # 未知标签用蓝色

        plt.scatter(x, y, c=color, marker=marker, s=size, label=label, alpha=0.7, edgecolors='w', linewidth=0.5)

    # 5. 图表美化
    plt.title("PCA Visualization of Embeddings (384D -> 2D)", fontsize=15)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Classes")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # 保存图片并显示
    output_image = "../outputs/embeddings_data/"+file_type+"_embedding_pca_visualization.png"
    plt.savefig(output_image, dpi=150)
    print(f"📊 散点图已保存为: {output_image}")
    plt.show()



if __name__ == "__main__":
    '''
        # 中文模型 'BAAI/bge-small-zh-v1.5' 或 'shibing624/text2vec-base-chinese'
        MODEL_NAME = "all-MiniLM-L6-v2"
    
        # 处理训练集
        generate_and_save_embeddings(
            parquet_path="../datasets/promptInjection/train.parquet",
            output_npy_path="../outputs/embeddings_data/train_embeddings.npy",
            model_name=MODEL_NAME
        )
    
        # 处理测试集
        generate_and_save_embeddings(
            parquet_path="../datasets/promptInjection/test.parquet",
            output_npy_path="../outputs/embeddings_data/test_embeddings.npy",
            model_name=MODEL_NAME
        )
    '''

    '''
    print("*" * 30)
    # 再次打印并查看基本信息
    train_data = np.load('../outputs/embeddings_data/train_embeddings.npy')
    print(f"查看部分结果：{train_data[:3]}")  # 打印数组内容
    print(f"数组形状: {train_data.shape}")  # 打印数组的维度信息
    print(f"数组数据类型: {train_data.dtype}") # 打印数组数据类型
    print(f"数组内存使用情况: {train_data.nbytes}")

    print(f"NaN值数量: {np.isnan(train_data).sum()}") # 打印数组中 NaN 的数量
    print(f"最小值：{train_data.min()}")
    print(f"最大值：{train_data.max()}")
    print(f"平均值：{train_data.mean()}")

    print("*" * 30)

    test_data = np.load('../outputs/embeddings_data/test_embeddings.npy')
    print(f"查看部分结果：{test_data[:3]}")  # 打印数组内容
    print(f"数组形状: {test_data.shape}")  # 打印数组的维度信息
    print(f"数组数据类型:{test_data.dtype}") # 打印数组数据类型
    print(f"数组内存使用情况:{test_data.nbytes}")

    print(f"NaN值数量: {np.isnan(test_data).sum()}") # 打印数组中 NaN 的数量
    print(f"最小值：{test_data.min()}")
    print(f"最大值：{test_data.max()}")
    print(f"平均值：{test_data.mean()}")

    print("=" * 30)
    '''



    # 如果你的标签列叫其他名字（如 'category', 'class'），请在这里修改
    LABEL_COL = "label"
    # 分别可视化训练集和测试集
    visualize_embeddings_2d(
        parquet_path="../datasets/promptInjection/train.parquet",
        npy_path="../outputs/embeddings_data/train_embeddings.npy",
        file_type = "train",
        label_column=LABEL_COL
    )

    visualize_embeddings_2d(
        parquet_path="../datasets/promptInjection/test.parquet",
        npy_path="../outputs/embeddings_data/test_embeddings.npy",
        file_type = "test",
        label_column=LABEL_COL
    )



"""
💡 代码亮点与使用建议：
自动处理空值：使用 fillna("") 防止文本列中存在 NaN 导致模型报错。
进度条反馈：生成 Embedding 通常比较耗时，show_progress_bar=True 能让你清楚看到处理进度。
模型选择：
如果你的数据是英文，默认的 all-MiniLM-L6-v2 是一个轻量且高效的选择（384维）。
如果你的数据是中文，建议将 MODEL_NAME 替换为 BAAI/bge-small-zh-v1.5 或 shibing624/text2vec-base-chinese，它们在中文语义检索上表现更好。
内存优化：生成的 .npy 文件可以直接被 FAISS 或 Qdrant 等向量数据库读取，用于后续的相似度检索或 RAG 系统构建。

Embedding 生成好后，要不要我帮你写一段用 FAISS 做向量相似度检索的代码？
"""