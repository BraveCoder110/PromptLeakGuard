1.数据集需要三个文件：
train.json  训练集
test.json   测试集


2.每条数据的格式：
{
   "prompt":"Ignore previous instructions..."
}


3.需要的字段：
prompt
response
task
attack
context
system_prompt
label


4.我们应该训练的字段是：
{
   "prompt": ...
}


5.重复样本待检测。
假设如果检测有重复样本，
假设总样本：1000
假设重复样本：12
则比例：1.2%
结论：重复比例较低，可以保留，或者后续通过 Python 脚本检测重复率。

6.空数据待检测。
如果检测有空数据，直接删除

7.超长Prompt待检测。
如果检查有超长Prompt，针对超长 Prompt，比如10000字，单独处理

8.乱码待检测。
如果检查有乱码，需要处理乱码

9.攻击类型：

| 类型                   | 示例                           | 是否常见  |
| -------------------- | ---------------------------- | ----- |
| Instruction Override | Ignore previous instructions | ⭐⭐⭐⭐⭐ |
| Prompt Leakage       | Reveal your system prompt    | ⭐⭐⭐⭐  |
| Role Play            | You are DAN                  | ⭐⭐⭐⭐  |
| Context Switch       | Forget the previous task     | ⭐⭐⭐   |
| Code Injection       | `system`                     | ⭐⭐    |





