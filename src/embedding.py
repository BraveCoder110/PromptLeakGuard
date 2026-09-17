"""
Generate embeddings

Use the all-MiniLM-L6-v2 model as the default model (baseline).
"""
import sys
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def generate_and_save_embeddings(parquet_path, output_npy_path, text_column="text", model_name="all-MiniLM-L6-v2"):
    """
    Read a Parquet file, generate text embeddings, and save them as a .npy file.
    """
    # 1. Load the embedding model
    # The model will be automatically downloaded from HuggingFace upon the first run
    print(f"Loading model: {model_name} ...")
    model = SentenceTransformer(model_name)

    # 2. Read Parquet data
    print(f"Reading file: {parquet_path} ...")
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"Failed to read Parquet file.: {e}")
        return

    if text_column not in df.columns:
        print(f"Error: Specified text column '{text_column}' not found. Current column names: {list(df.columns)}")
        return

    print(f"Print the DataFrame to inspect the data.:{df.head()}")

    # Extract the list of text and handle any potential null values.
    texts = df[text_column].astype(str).fillna("").tolist()
    print(f"Successfully read {len(texts)} records; starting embedding generation...")

    # 3. Batch generate embedding vectors
    # show_progress_bar=True # Displays a progress bar, making it easier to monitor time-consuming tasks
    embeddings = model.encode(
        texts,
        batch_size=32,            # Can be adjusted based on VRAM/RAM size.
        show_progress_bar=True,
        convert_to_numpy=True     # Directly output in NumPy array format.
    )

    # 4. Save as a .npy file.
    print(f"save Embedding to: {output_npy_path} ...")
    np.save(output_npy_path, embeddings)
    print(f"success,The vector shape is: {embeddings.shape}")



def visualize_embeddings_2d(parquet_path, npy_path, file_type, label_column="label", text_column="text"):
    """
    Read the labels from the Parquet file and the embeddings from the .npy file, reduce the dimensionality to two using PCA, and generate a scatter plot.
    """
    # 1. loading data
    print(f"loading: {parquet_path} and {npy_path} ...")
    try:
        df = pd.read_parquet(parquet_path)
        embeddings = np.load(npy_path)
    except Exception as e:
        print(f"failed: {e}")
        return

    # 2. Check if the label column exists.
    if label_column not in df.columns:
        print(f"error：not found '{label_column}'。current column: {list(df.columns)}")
        return

    # 3. PCA dimensionality reduction (384 dimensions -> 2 dimensions)
    print("Performing PCA dimensionality reduction (384D -> 2D)...")
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    print(f"PCA dimensionality reduction complete. Proportion of explained variance.: {pca.explained_variance_ratio_}")

    # 4. Scatter plot
    plt.figure(figsize=(10, 8))

    # Get all unique tags
    unique_labels = df[label_column].unique()

    # print(unique_labels)
    # sys.exit(0)


    colors = {"Attack": "#e74c3c", "Safe": "#2ecc71"}  # Use red for attacks and green for safety.

    for label in unique_labels:
        # Filter out the 2D coordinates belonging to the current label.
        mask = (df[label_column] == label)
        x = embeddings_2d[mask, 0]
        y = embeddings_2d[mask, 1]

        # Compatible with both numeric and string labels
        # Assumes 1 or 'attack' represents the attack class, and 0 or 'safe' represents the safe class
        if str(label).lower() in ["1", "attack"]:
            marker = '.'
            size = 50
            color = "#e74c3c"  # red attack
        elif str(label).lower() in ["0", "safe"]:
            marker = '^'
            size = 60
            color = "#2ecc71"  # green safe
        else:
            marker = 'o'  # other cycle
            size = 50
            color = "#3498db"  # unknown  blue

        plt.scatter(x, y, c=color, marker=marker, s=size, label=label, alpha=0.7, edgecolors='w', linewidth=0.5)

    # 5. graph
    plt.title("PCA Visualization of Embeddings (384D -> 2D)", fontsize=15)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Classes")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    output_image = "../outputs/embeddings_data/"+file_type+"_embedding_pca_visualization.png"
    plt.savefig(output_image, dpi=150)
    print(f" The scatter plot has been saved as: {output_image}")
    plt.show()



if __name__ == "__main__":
    '''
        # generate Embedding，中文模型 'BAAI/bge-small-zh-v1.5' 或 'shibing624/text2vec-base-chinese'
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
    # check basic info
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



    # 降维并绘制散点图，Perform dimensionality reduction and plot a scatter plot; if your label column has a different name (e.g., 'category', 'class'), please modify it here.
    LABEL_COL = "label"
    # train
    visualize_embeddings_2d(
        parquet_path="../datasets/promptInjection/train.parquet",
        npy_path="../outputs/embeddings_data/train_embeddings.npy",
        file_type = "train",
        label_column=LABEL_COL
    )

    #test
    visualize_embeddings_2d(
        parquet_path="../datasets/promptInjection/test.parquet",
        npy_path="../outputs/embeddings_data/test_embeddings.npy",
        file_type = "test",
        label_column=LABEL_COL
    )