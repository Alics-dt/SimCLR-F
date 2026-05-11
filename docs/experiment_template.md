# 实验设置模板

## 1. 数据集
- **主数据集**：CIFAR-10 / CIFAR-100
- **扩展数据集**：STL-10, Tiny-ImageNet 或 ImageNet-100
- **鲁棒性数据集**：CIFAR-10-C（如可用）

## 2. 预处理与增强
- RandomResizedCrop(size=32 or 96)
- ColorJitter
- RandomGrayscale
- GaussianBlur（按分辨率开关）
- HorizontalFlip

## 3. 模型结构
- Backbone: ResNet-18（起步）/ ResNet-50（完整版）
- Projection Head: MLP(2048 -> 2048 -> 128)
- 输出归一化：L2 normalize

## 4. 训练超参数（默认）
- Epochs: 200（快速调试可设1~20）
- Batch size: 256
- Optimizer: SGD(momentum=0.9, weight_decay=1e-4)
- LR: 0.3（按batch线性缩放）
- Scheduler: cosine decay
- Temperature τ: 0.2
- hard weight系数 α: 1.0
- reliability系数 β: 1.0
- kNN估计k: 10

## 5. 对比方法
- SimCLR baseline
- Hard-negative only
- Reliability-only
- RAHNCL (hard + reliability)

## 6. 评价指标
- 线性评估Top-1准确率
- kNN准确率
- 训练稳定性：loss均值与方差
- OOD指标：mCE / OOD Top-1（可选）
- 统计显著性：3个随机种子平均±标准差

## 7. 结果报告模板
- 表1：主数据集Top-1比较
- 表2：消融实验
- 图1：训练曲线
- 图2：鲁棒性/OOD对比
