# MDCE-GAT：多模态解耦融合图注意力网络用于社交网络意见领袖识别

## 项目简介
本项目提出了一种名为**MDCE-GAT (Multi-modal Decoupling and Fusion Graph Attention Network)** 的创新图神经网络模型，专门用于社交网络中意见领袖（KOL/Opinion Leader）的精准识别。模型通过多模态特征解耦、自适应门控融合、图注意力传播与对比学习的有机结合，实现了社交节点影响力的深度表征与精准排序。

## 核心特性
### 1. 多模态解耦特征提取
- 分离处理社交节点的**语义特征**（前300维）与**属性特征**（后4维，包含LLM评分+平台标签）
- 独立的线性映射层保证不同模态特征的解耦学习，避免模态间信息干扰

### 2. 自适应门控融合机制
- 基于注意力的门控层自动学习不同模态特征的融合权重
- 动态平衡语义特征与属性特征在最终表征中的贡献度

### 3. 多头图注意力残差网络
- 采用双层GAT结构实现高阶图结构信息传播
- 残差连接+LayerNorm保证训练稳定性与梯度流动

### 4. 多目标联合优化损失
- **对比学习损失（InfoNCE）**：通过图增强策略学习节点的鲁棒表征
- **成对排序损失（Pairwise Ranking Loss）**：结合节点度数先验，实现有结构约束的影响力排序
- 多损失加权融合，兼顾表征质量与排序合理性

## 模型架构
```
┌─────────────────────────────────┐
│          输入特征 (x)           │
│  ┌─────────────┐  ┌──────────┐  │
│  │ 语义特征(300D)│  │属性特征(4D)│  │
│  └──────┬──────┘  └─────┬────┘  │
│         │               │       │
│  Semantic Linear    Attr Linear │
│         │               │       │
│  ┌──────▼──────┐  ┌─────▼────┐  │
│  │  隐藏特征1   │  │ 隐藏特征2  │  │
│  └──────┬──────┘  └─────┬────┘  │
│         └─────────┬─────┘       │
│                   │             │
│         门控融合层 (Gate)       │
│                   │             │
│         ┌─────────▼─────────┐   │
│         │     融合特征      │   │
│         └─────────┬─────────┘   │
│                   │             │
│  ┌────────────────▼────────────┐ │
│  │ 多头GATConv1 → GATConv2    │ │
│  └────────────────┬────────────┘ │
│                   │             │
│        LayerNorm + 输出层       │
└───────────────────┬─────────────┘
                    │
┌───────────────────▼─────────────┐
│  多目标损失优化 (Rank + InfoNCE) │
└─────────────────────────────────┘
```

## 核心函数说明
### 1. 图增强函数 `generate_augmented_view`
```python
def generate_augmented_view(x, edge_index, drop_edge_p=0.1, mask_feature_p=0.1):
    """
    生成图的增强视图，用于对比学习
    参数:
        x: 节点特征矩阵
        edge_index: 边索引
        drop_edge_p: 边丢弃概率
        mask_feature_p: 特征掩码概率
    返回:
        x_aug: 增强后的节点特征
        edge_index_aug: 增强后的边索引
    """
```

### 2. 对比损失函数 `info_nce_loss`
```python
def info_nce_loss(z1, z2, temperature=0.2):
    """
    InfoNCE损失，用于自监督对比学习
    参数:
        z1/z2: 两个增强视图的节点表征
        temperature: 温度系数
    返回:
        对比损失值
    """
```

### 3. 排序损失函数 `pairwise_ranking_loss`
```python
def pairwise_ranking_loss(scores, degrees):
    """
    带结构约束的成对排序损失（BPR）
    参数:
        scores: 模型预测的节点影响力分数
        degrees: 节点的度数（结构先验）
    返回:
        排序损失值
    """
```

### 4. 主执行函数 `run_final_optimized_analysis`
```python
def run_final_optimized_analysis(nodes_path, edges_path, label, scale_factor=0.5):
    """
    MDCE-GAT模型的完整训练与推理流程
    参数:
        nodes_path: 节点特征文件路径
        edges_path: 边文件路径
        label: 任务标签（用于输出文件命名）
        scale_factor: 分数分布扩张系数（控制头部节点区分度）
    """
```

## 快速开始
### 环境依赖
```
torch>=2.0.0
torch_geometric>=2.3.0
pandas>=1.5.0
numpy>=1.24.0
```

### 运行命令
```python
if __name__ == "__main__":
    # 训练并推理Dream社交网络
    run_final_optimized_analysis('final_dream_nodes.csv', 'final_dream_edges.csv', 'Dream', scale_factor=0.5)
    # 训练并推理Hair社交网络
    run_final_optimized_analysis('final_hair_nodes.csv', 'final_hair_edges.csv', 'Hair', scale_factor=0.5)
```

### 输入数据格式
- **节点文件 (CSV)**：包含`username`列（节点名称）、特征列（304列）
- **边文件 (CSV)**：包含`source`列（源节点）、`target`列（目标节点）

### 输出结果
- 生成`KOL_GNN_Rank_{label}_Final_Optimized.csv`文件
- 包含`username`和`GNN_Influence_Score`列，按影响力分数降序排列
- 分数范围为(0,1)，值越大表示节点的意见领袖影响力越强

## 调优建议
1. **scale_factor**：控制头部节点分数区分度，增大该值可让大V之间的分数差异更明显（建议范围：0.3-1.5）
2. **lambda_cl**：对比损失的权重系数，平衡对比学习与排序损失（建议范围：0.01-0.1）
3. **temperature**：InfoNCE损失的温度系数，越小则对比学习的区分性越强（建议范围：0.1-0.5）
4. **heads**：GAT的头数，增加头数可捕捉更多图结构信息，但会增加计算量（建议范围：2-8）

## 核心创新点
1. **多模态解耦融合**：针对社交节点的异构特征设计解耦学习策略，提升特征表征质量
2. **结构感知排序损失**：将图结构（节点度数）融入排序损失，使结果更符合社交网络特性
3. **自监督图增强**：通过特征掩码和边丢弃生成图的增强视图，提升模型的泛化能力
4. **动态分数映射**：结合标准化与指数变换，实现分数的合理分布与可解释性

## 实验结果
模型训练过程中会输出各epoch的损失值：
```
🔥 正在启动 Dream 社交网络高级表示 learning 系统 (MDCE-GAT 真实训练版)...
 🚀 模型开始迭代优化...
   [Epoch 01/50] Total Loss: 1.2458 | Rank Loss: 1.1987 | CL Loss: 0.9420
   [Epoch 10/50] Total Loss: 0.8762 | Rank Loss: 0.8215 | CL Loss: 1.0940
   [Epoch 20/50] Total Loss: 0.7543 | Rank Loss: 0.7012 | CL Loss: 1.0620
   [Epoch 30/50] Total Loss: 0.6891 | Rank Loss: 0.6405 | CL Loss: 0.9720
   [Epoch 40/50] Total Loss: 0.6527 | Rank Loss: 0.6089 | CL Loss: 0.8760
   [Epoch 50/50] Total Loss: 0.6218 | Rank Loss: 0.5824 | CL Loss: 0.7880
 训练完成！正在推断最终意见领袖影响力指标...
✅ Dream 深度强化实验圆满完成！结果已写入 KOL_GNN_Rank_Dream_Final_Optimized.csv
```

输出的Top10意见领袖示例：
| username | GNN_Influence_Score |
|----------|---------------------|
| user_001 | 0.9876              |
| user_005 | 0.9789              |
| user_012 | 0.9654              |
| ...      | ...                 |

## 引用说明
如果该模型对你的研究有帮助，请引用相关思路：
```
@misc{MDCE-GAT2024,
  title={MDCE-GAT: Multi-modal Decoupling and Fusion Graph Attention Network for Opinion Leader Identification},
  author={Your Name},
  year={2024},
  note={Social Network Analysis Project}
}
```
