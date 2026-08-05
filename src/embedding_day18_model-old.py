"""
生成embedding

使Day18任务三：Embedding空间检查

将day17：./docs/day17_instruction_bias.csv

"""
import sys
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import chardet

def generate_and_save_embeddings(parquet_path, output_npy_path, text_column="text", model_name="all-MiniLM-L6-v2"):
    """
    读取 parquet 文件，生成文本 Embedding 并保存为 .npy 文件。
    """
    '''
        #检测编码
        with open(parquet_path, "rb") as f:
            raw = f.read(100000)   # 读取前100KB即可
        result = chardet.detect(raw)
        print("-----",result)
    '''

    # 1. 加载 Embedding 模型
    # 首次运行会自动从 HuggingFace 下载模型
    print(f"正在加载模型: {model_name} ...")
    model = SentenceTransformer(model_name)

    # 2. 读取 Parquet 数据
    print(f"正在读取文件: {parquet_path} ...")
    try:
        df = pd.read_csv(parquet_path, encoding="gb18030")
    except Exception as e:
        print(f"读取 csv 文件失败: {e}")
        return

    # 提取文本列表，并处理可能存在的空值
    texts = df["text"].tolist()
    print(f"成功读取 {len(texts)} 条数据，开始生成 Embedding...")
    print(f"查看数据：{texts}")

    # 3. 批量生成 Embedding 向量
    # show_progress_bar=True #会显示进度条，方便观察长耗时任务
    embeddings = model.encode(
        texts,
        batch_size=32,            # 可根据显存/内存大小调整
        show_progress_bar=True,
        convert_to_numpy=True     # 直接输出 numpy 数组格式
    )

    print(f"----:{embeddings}---")

    # 4. 保存为 .npy 文件
    print(f"正在保存 Embedding 到: {output_npy_path} ...")
    np.save(output_npy_path, embeddings)
    print(f"✅ 成功! 向量形状为: {embeddings.shape}")


if __name__ == "__main__":

    # BGE-M3 模型
    MODEL_NAME = "BAAI/bge-m3"
    file_name = "bgeM3"

    # 处理训练集
    generate_and_save_embeddings(
        parquet_path="../docs/day17_instruction_bias.csv",
        output_npy_path="../outputs/embeddings_data/day17_embeddings_"+file_name+".npy",
        model_name=MODEL_NAME
    )