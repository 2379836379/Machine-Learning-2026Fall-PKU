# 数据文件

| 文件 | 用途 | 数据行数（不含表头） |
| --- | --- | ---: |
| `full_train.csv` | 构建词表并训练模型 | 67,349 |
| `full_test.csv` | 最终预测与评估 | 872 |

CSV 均有 `text,label` 两列。例如，读取一行 `"A wonderful and beautiful film",1`，输入文本是 `A wonderful and beautiful film`，标签输出为 `1`（positive）；`0` 表示 negative。实际文件可以用 `pandas.read_csv` 读取。

完整数据来自 SST-2：`full_train.csv` 是原始 train split，`full_test.csv` 是完整的有标签 dev split。词表和模型只能从 `full_train.csv` 构建；`full_test.csv` 的文本只用于预测，标签只用于计算准确率。
