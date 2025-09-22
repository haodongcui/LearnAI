## MAML
与模型无关的元学习（Model-Agnostic Meta-Learning）

### 思想
学会如何学习\
$\implies$学会学习技巧\
$\implies$少数次学习就可以学好\
$\implies$快速适应能力

### 流程
1. 数据集
    - 训练集（包括支持集和查询集）
    - 测试集（包括支持集和查询集）
2. 内部模型
3. MAML算法：
    - 对每个任务
        - 复制当前模型参数
        - 内循环（支持集上正常训练：计算损失、反向传播、更新内部参数）
        - 外循环预备（在查询集上计算损失，计算梯度，保存/累积梯度）
    - 外循环（更新模型参数）
    - 测试：在测试集的支持集上训练，在测试集的查询集上测试，打印结果（loss和accs）
    - 一个epoch结束，再重复整个算法


### 主要参考：
- MAML原论文 https://arxiv.org/abs/1703.03400
- 注释齐全，适合入门的代码 https://github.com/daetz-coder/Pytorch-MAML-Tutorial
- 内外循环之间的参数转移操作 https://blog.51cto.com/u_16213358/12689649

