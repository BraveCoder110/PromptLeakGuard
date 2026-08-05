
文件目录

PromptLeakGuard/
│
├── datasets/               # 存放数据集
│   ├── promptInjection/    # 存放原始数据集
│   ├── processed/          # 存放处理后的数据集
│
├── src/
│   ├── preprocessing.py    # 数据预处理脚本
│   ├── embedding.py        # 嵌入生成脚本
│   ├── train.py            # 训练脚本
│   ├── evaluate.py         # 评估脚本  
│
├── models/
│
├── outputs/
│
├── docs/
│
└── README.md

---

论文题目：

PromptLeakGuard: A Lightweight Explainable Semantic Risk Detection Framework for Prompt Injection Attacks in Large Language Models



PromptLeakGuard：一种面向未知Prompt Injection攻击的轻量级语义风险评估框架。  -》》》 Risk Assessment


v1.3：
PromptLeakGuard: An Explainable Lightweight Semantic Risk Assessment Framework for Prompt Injection Detection
PromptLeakGuard：一种面向 Prompt Injection 的可解释轻量级语义风险评估框架




Input Normalization：

全角。

半角。

统一。

不会改变语义。

ＡＢＣ

↓

ABC


空白字符规范化，保留：

语义。

Hello


World

删除\t、\r、\n等





1.论文Dataset里可以写：

The PromptInject dataset consists of only two fields (text and label), making it directly suitable for semantic embedding without manual feature engineering.
这句话以后直接放论文 Dataset 一节。



2.论文：

Dataset。

可以写：

Although the training set is mildly imbalanced, the imbalance ratio remains acceptable for standard supervised classifiers.

3.
dataset增加以下图：

Dataset Profile

Samples

662

↓

Train

546

↓

Test

116

↓

Safe

399

↓

Attack

263


---

4.论文dataset章节：

直接：

写：

We set the maximum input length to 256 tokens according to the dataset length distribution.


5.dataset章节：
| 属性             |              数值 |
| -------------- | --------------: |
| Train Samples  |             546 |
| Test Samples   |             116 |
| Total Samples  |             662 |
| Safe Samples   |             399 |
| Attack Samples |             263 |
| Languages      | English, German |
| Avg Length     |       118 chars |
| Median Length  |        63 chars |
| Max Length     |      4545 chars |
| Missing Values |               0 |



6.以后论文 Discussion 可以写：
Sentence-level embedding may dilute localized malicious instructions when benign and malicious contents coexist in the same prompt.


7.模型目前：

更擅长识别"显式攻击"，不擅长识别"隐式攻击"。


8.第六部分：我建议把 Error Analysis 写成论文里的一个表
| Error Category       | Possible Cause                          | Potential Improvement  |
| -------------------- | --------------------------------------- | ---------------------- |
| Short Prompt         | Limited semantic information            | Context augmentation   |
| Hidden Injection     | Benign context dilutes malicious intent | Chunk-level encoding   |
| Cross-language       | English-centric encoder                 | Multilingual embedding |
| Task-oriented Prompt | Similar to benign requests              | Hard negative training |
| Mixed Prompt         | Global embedding loses local attack     | Hierarchical encoding  |






插入内容：
以下是我们的baseline，用all-MiniLM-L6-v2 lg训练的
结果如下： 🔄 正在加载 Embedding 和标签数据... ✅ 训练集形状: (546, 384), 测试集形状: (116, 384) 🤖 正在训练 Logistic Regression 模型... ✅ 模型训练完成！ 🔮 正在生成预测结果... ======================================== 📊 模型评估指标 (Test Set) ======================================== 🎯 Accuracy: 0.8276 🎯 Precision: 0.8597 🎯 Recall: 0.8327 🎯 F1-score: 0.8250 ======================================== 📊 混淆矩阵已保存为: lr_confusion_matrix.png
TN = 55
FP = 1
FN = 19
TP = 41


9.科研假设Hypothesis H1：采用更强的多语言语义表示，可以显著减少跨语言 Prompt Injection 的 False Negative。
结果：
到目前为止，我们实际上已经完成了一个非常规范的 Embedding Comparison Experiment。
| Encoder               |     TN |    FP |    FN |     TP |
| --------------------- | -----: | ----: | ----: | -----: |
| all-MiniLM-L6-v2      |     55 |     1 |    19 |     41 |
| multilingual-e5-small |     56 |     0 |    16 |     44 |
| BGE-small-en          |     56 |     0 |    15 |     45 |
| **BGE-M3**            | **56** | **0** | **7** | **53** |


第一大发现：
我们当初提出了一个科研假设（Hypothesis）：

更好的语义表示是否能够减少 False Negative？

现在实验已经回答了。

答案是：

可以，而且效果非常明显。

看看 FN：
| Encoder         | False Negative |
| --------------- | -------------: |
| MiniLM          |             19 |
| multilingual-e5 |             16 |
| BGE-small       |             15 |
| **BGE-M3**      |          **7** |



False Negative 减少了 63.2%。
(19 - 7) / 19 = 63.2%


这说明：真正影响性能的不是 Logistic Regression，而是语义表示（Embedding）。



第二大发现：
看看 False Positive。
| Encoder         | FP |
| --------------- | -: |
| MiniLM          |  1 |
| multilingual-e5 |  0 |
| BGE-small       |  0 |
| BGE-M3          |  0 |


非常漂亮。

说明：

提高 Recall 的同时，

并没有增加误报。

这是很多模型做不到的。

通常：

Recall 提高。

Precision 会下降。

而你的实验没有出现这个问题。

这是一个很好的现象。


第三大发现：
第四大发现（我最兴奋的一点）

其实。

我们已经可以回答论文里的 Research Question。

例如：

论文：

可以写：

RQ1: Does multilingual semantic representation improve prompt injection detection?

实验：

回答：

Yes.

证据：

FN：

19

↓

7

FP：

保持：

0。

这就是：

Research Question。










比较：
| Encoder               |   Accuracy |  Precision |     Recall |         F1 |    FN |    FP |
| --------------------- | ---------: | ---------: | ---------: | ---------: | ----: | ----: |
| all-MiniLM-L6-v2      |     0.8276 |     0.8597 |     0.8327 |     0.8250 |    19 |     1 |
| multilingual-e5-small |     0.8621 |     0.8889 |     0.8667 |     0.8606 |    16 |     0 |
| BGE-small-en          |     0.8707 |     0.8944 |     0.8750 |     0.8695 |    15 |     0 |
| **BGE-M3**            | **0.9397** | **0.9444** | **0.9417** | **0.9396** | **7** | **0** |

BGE-M3 全面领先。





train_model.py
output:
C:\Users\liyoubao\AppData\Local\Python\bin\python.exe C:\Users\liyoubao\Desktop\MScAI-Project\PromptLeakGuard\src\train_model.py
正在加载真实数据集...
数据加载完成: Train=(546, 384), Test=(116, 384)
正在加载模型: BAAI/bge-m3 ...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|██████████| 391/391 [00:00<00:00, 24177.69it/s]
正在生成带 L2 归一化的 Embedding...
Batches: 100%|██████████| 18/18 [02:11<00:00,  7.29s/it]
Batches: 100%|██████████| 4/4 [00:15<00:00,  3.93s/it]
正在训练 Logistic Regression...

模型评估报告 (Test Set):
precision    recall  f1-score   support

        Safe       0.89      1.00      0.94        56
      Attack       1.00      0.88      0.94        60

    accuracy                           0.94       116
macro avg       0.94      0.94      0.94       116
weighted avg       0.95      0.94      0.94       116

模型已保存至: ./models/bge_m3_lr.joblib

进程已结束，退出代码为 0








