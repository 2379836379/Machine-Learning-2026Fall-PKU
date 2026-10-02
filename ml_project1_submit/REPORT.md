# Project 1 实验报告

## 运行与结果

运行命令:
```python
uv sync
uv run python main.py
```
测试准确率：0.8142  
运行结果:
```
Training examples: 67349
Test examples: 872
Vocabulary size: 5000
Test accuracy: 0.8142
Most positive words:
touching 4.256
playful 4.198
wonderfully 4.085
detailed 4.044
heartwarming 4.016
mesmerizing 4.002
refreshingly 3.987
stunning 3.972
pleasing 3.972
riveting 3.972
Most negative words:
unfunny -4.960
poorly -4.886
pointless -4.776
tiresome -4.576
badly -4.563
unnecessary -4.509
suffers -4.495
ugly -4.466
inept -4.437
listless -4.375
```

## 1.情感词分析

正向倾向的10个词：`touching`、`playful`、`wonderfully`、`detailed`、`heartwarming`、`mesmerizing`、`refreshingly`、`stunning`、`pleasing`、`riveting`。

负向倾向的10个词：`unfunny`、`poorly`、`pointless`、`tiresome`、`badly`、`unnecessary`、`suffers`、`ugly`、`inept`、`listless`。

举例：`heartwarming` 和 `stunning` 明显表达正面评价，`unfunny` 和 `pointless` 明显表达负面评价，符合直觉。

## 2.Bad cases

部分误分类样例：
1. `we root for ... though perhaps it's an emotion closer to pity.`（真实正类，预测负类）：句子包含 `pity` 等负面词，但整体语义仍是支持角色。
2. `a bad high school production ... without benefit of song.`（真实负类，预测正类）：局部上下文中的常见词可能抵消了 `bad` 的负面信号。
3. `it's a cookie-cutter movie, a cut-and-paste job.`（真实负类，预测正类）：讽刺和复合短语的语义无法由独立词特征充分表达。

这些错误反映了词袋表示忽略词序、否定范围和词语之间的组合关系。

## 3.词表大小对比

| 词表大小 | 测试准确率 |
| ---: | ---: |
| 1,000 | 0.7569 |
| 2,000 | 0.7913 |
| 5,000 | 0.8142 |
| 10,000 | 0.8062 |

结果说明词表大小和准确率不是简单的单调关系，而是存在一个较合适的范围。1,000过小，只保留了最高频词，许多具有区分度的情感词被丢弃，模型表达能力不足。2,000时，增加了更多情感相关词，模型获得了更多有效信息，因此准确率明显提升。5,000时准确率进一步提升。10,000时，新增的词大多是低频词，可能存在以下问题： 统计次数少，条件概率估计不稳定；朴素贝叶斯把词当作相互独立，低频噪声会累积影响分类。因此，词表太小时模型会欠拟合，太大时可能引入噪声甚至过拟合。

## 4.特征表示改进实验

### 方法

原始 B 部分只为每条文本生成 unigram 特征。为利用局部词序信息，在 `src/preprocessing.py` 中增加了可选的 bigram 特征：相邻词组成形如 `not_good`、`very_funny` 的新 token，并与 unigram 一起构成 Bernoulli 0/1 特征。词表仍然只从训练集构建，C 部分的加一平滑和预测公式保持不变。程序中的 `NGRAM_RANGE = (1, 2)` 开启 unigram + bigram；改回 `(1, 1)` 即可复现原始基线。

### 实验结果

在词表大小固定为 5,000 时，使用同一份训练集和测试集得到：

| 特征表示 | 词表大小 | 测试准确率 |
| --- | ---: | ---: |
| unigram（原始） | 5,000 | 0.8142 |
| unigram + bigram | 5,000 | 0.8108 |

在 unigram + bigram 特征固定不变的条件下调整词表大小：

| 词表大小 | 测试准确率 |
| ---: | ---: |
| 1,000 | 0.7661 |
| 2,000 | 0.7741 |
| 3,000 | 0.7856 |
| 5,000 | 0.8108 |

### 分析

加入 bigram 后准确率从 0.8142 降至 0.8108，没有带来提升。原因可能是词表大小仍固定为 5,000，bigram 占用了原本可以保留更多高频 unigram 的位置；许多 bigram 出现次数较少，经过加一平滑后概率估计不稳定。与此同时，朴素贝叶斯仍假设各特征条件独立，相关的 unigram 与 bigram 可能造成重复计权。这个结果说明特征更丰富不一定更好，特征类型和词表大小需要一起调节；bigram 对否定和短语的理论优势，在当前固定词表和数据规模下被低频噪声抵消了。综合本次实验，当前数据和模型下仍应优先选择原始 unigram + 5,000 词表的配置。
