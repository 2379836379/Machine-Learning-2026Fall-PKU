# Project 1 实验报告

## 运行与结果

运行命令：`/tmp/ml_project1_venv2/bin/python main.py`（安装 `numpy`、`pandas` 后也可直接运行 `python main.py`）。

- 训练集：67,349 条
- 测试集：872 条
- 词表大小：5,000
- 测试准确率：0.8142

模型使用 Bernoulli 0/1 特征和加一平滑，并在对数空间计算预测分数。

## 情感词

正向倾向较强的词包括：`touching`、`playful`、`wonderfully`、`detailed`、`heartwarming`、`mesmerizing`、`refreshingly`、`stunning`、`pleasing`、`riveting`。

负向倾向较强的词包括：`unfunny`、`poorly`、`pointless`、`tiresome`、`badly`、`unnecessary`、`suffers`、`ugly`、`inept`、`listless`。

例如 `heartwarming` 和 `stunning` 明显表达正面评价，`unfunny` 和 `pointless` 明显表达负面评价，符合直觉。

## Bad cases

部分误分类样例：

1. `we root for ... though perhaps it's an emotion closer to pity.`（真实正类，预测负类）：句子包含 `pity` 等负面词，但整体语义仍是支持角色。
2. `a bad high school production ... without benefit of song.`（真实负类，预测正类）：局部上下文中的常见词可能抵消了 `bad` 的负面信号。
3. `it's a cookie-cutter movie, a cut-and-paste job.`（真实负类，预测正类）：讽刺和复合短语的语义无法由独立词特征充分表达。

这些错误反映了词袋表示忽略词序、否定范围和词语之间的组合关系。

## 词表大小对比

| 词表大小 | 测试准确率 |
| ---: | ---: |
| 1,000 | 0.7569 |
| 5,000 | 0.8142 |

词表从 1,000 增加到 5,000 后，更多具有区分度的情感词被保留，准确率明显提升；继续增大词表需要更多内存，也可能引入低频噪声。加一平滑能避免未见特征造成零概率。
