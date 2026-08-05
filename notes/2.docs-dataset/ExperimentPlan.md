## 一、Baseline Experiment（基线实验）
Sentence-BERT

↓

Logistic Regression

check accuarcy


## 二、Comparison Experiment（对比实验），比较方法比别人好吗？
| Encoder | Classifier |
| ------- | ---------- |
| SBERT   | LR         |
| SBERT   | SVM        |
| SBERT   | RF         |
| SBERT   | XGBoost    |



## 三、Ablation Study（消融实验），消融不是换算法，而是验证你的设计有没有必要
Normalization

↓

SBERT

↓

XGBoost

↓

Explainable Output


如：
实验1，没有Normalization看看性能。
实验2，没有Explainable看看速度。
实验3，换Encoder，例如MiniLM。



## 四、Generalization Experiment（泛化实验）

未知攻击。

如果还能检测说明泛化很好
再例如训练英文测试中文也是泛化，训练模板攻击测试真实用户也是泛化。

| 实验                    | 目的            |
| --------------------- | ------------- |
| Baseline              | 方法是否有效        |
| Classifier Comparison | 哪个分类器最好       |
| Encoder Comparison    | 哪个Embedding最好 |
| Ablation              | 每个模块是否有贡献     |
| Generalization        | 是否能识别未知攻击     |
| Efficiency            | 是否够快          |
| Case Study            | 展示典型成功/失败案例   |



## 五。总结：
| 实验编号 | 内容                               | 目的       |
| ---- | -------------------------------- | -------- |
| E1   | SBERT + LR                       | Baseline |
| E2   | SBERT + SVM                      | 分类器对比    |
| E3   | BGE + LR                         | 编码器对比    |
| E4   | 去掉 Input Normalization           | 消融实验     |
| E5   | PromptInject 训练 → TensorTrust 测试 | 泛化实验     |
| E6   | 推理时间、CPU占用                       | 效率实验     |
