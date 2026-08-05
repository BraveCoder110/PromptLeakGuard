以后论文的Introduction和Discussion

1.我们为什么研究Prompt Injection？ 此处在写的时候注意在chatgpt里在搜一遍
随着ChatGPT、DeepSeek等大语言模型被广泛应用，Prompt Injection（提示词注入）已成为最常见的安全威胁之一。本项目聚焦于输入安全检测这一关键环节，通过构建轻量级语义分析模型，对用户输入进行风险识别，并输出攻击类型、风险等级和检测结果，为大模型提供第一道安全防线。
identify potentially malicious prompts and provide an early security screening mechanism before user inputs are processed by the target LLM.

2.现有方法最大问题是什么？
Existing prompt injection detection methods mainly rely on keyword matching, handcrafted rules, or general semantic representations. Although these approaches can identify obvious attacks, they often struggle with multilingual prompts, semantically ambiguous instructions, and hidden malicious intents embedded within benign contexts. Consequently, false negatives remain a major challenge for practical deployment.


3.我们的Baseline是什么？为什么？
The baseline consists of a Sentence-BERT encoder followed by a Logistic Regression classifier. Logistic Regression was intentionally selected because it is lightweight, interpretable, and allows the influence of different semantic representations to be evaluated without introducing additional model complexity.

4.Baseline失败在哪里？Day12
语义模糊、有些指令嵌入正常请求、受多语言影响、还有些任务型 Prompt(比如Generate SQL)、长上下文攻击(正常内容+后面攻击)
These observations indicate that the primary limitation lies in semantic representation rather than the downstream classifier.

5.我们提出什么Hypothesis？
H1:
Replacing the baseline encoder with a stronger multilingual semantic encoder will significantly reduce False Negatives while maintaining a low False Positive rate.
while maintaining a low False Positive rate.


6.实验如何验证？
我们使用了三种方法来验证我们的Hypothesis：multilingual-e5-small、BGE-small-en、BGE-M3
To ensure a fair comparison, all encoders were evaluated using the same Logistic Regression classifier, identical training settings, and the same PromptInject dataset.


7.最终结论是什么？
BGE-M3的表现最好，Accuracy从82.76%提升到93.97%,提升了11.21%。Recall从83.27%提升到94.17%，提升了10.9%。 False Negative 减少了 63.2%。其次是BGE-small。
The experimental results support the proposed hypothesis, demonstrating that semantic representation quality has a greater impact on Prompt Injection detection performance than changing the downstream classifier alone.

