# MSc Project: Optimization Algorithms and Quantization Methods for Neural Networks

## 项目概述

本项目研究并实现了多种优化算法和量化方法，用于提高神经网络训练的效率和模型部署的性能。项目主要关注梯度下降变体、指数梯度方法以及各种量化技术（INT4、INT8、LNS）的实现和比较。

## 主要特性

### 🚀 优化算法
- **Custom SGD**: 支持标准梯度下降(GD)和指数梯度(EG)的增强SGD优化器
- **AdamWeg**: 结合Adam和指数梯度的混合优化器
- **AdamGD**: 结合Adam和梯度下降的混合优化器  
- **LNS_Madam**: 基于对数数系统的Madam优化器

### 🔧 量化方法
- **INT4量化**: 对称和非对称4位整数量化
- **INT8量化**: 8位整数量化
- **LNS量化**: 4位和8位对数数系统量化
- **混合精度**: 支持不同精度的混合量化策略

### 📊 模型架构
- **CNN模型**: 卷积神经网络用于图像分类
- **MLP模型**: 多层感知机模型
- **数据集**: MNIST和Fashion-MNIST

## 项目结构

```
MScProject/
├── custom_optim.py          # 自定义优化器实现
├── lns_quantizer_optimized.py  # LNS量化器
├── true_int4_quantizer.py   # INT4量化器
├── train.py                 # MNIST训练脚本
├── train_fashionmnist.py    # Fashion-MNIST训练脚本
├── CnnModel.py             # 模型架构定义
├── Data.py                 # 数据处理模块
├── tests/                  # 单元测试
├── plotscode/              # 绘图和分析代码
└── README.md               # 项目说明文档
```

## 安装和运行

### 环境要求
- Python 3.8+
- PyTorch 1.9+
- NumPy, Matplotlib, Seaborn
- CUDA支持（可选，用于GPU加速）

### 安装依赖
```bash
pip install torch torchvision numpy matplotlib seaborn pandas
```

### 运行训练
```bash
# MNIST数据集训练
python train.py --lr 0.01 --weight_decay 0.001

# Fashion-MNIST数据集训练  
python train_fashionmnist.py --lr 0.01 --weight_decay 0.001
```

### 运行量化
```bash
# INT4量化
python quantize_true_int4.py

# LNS量化
python quantize_lns_optimized.py
```

## 实验结果

项目包含详细的实验结果分析，包括：
- 不同优化器的训练曲线对比
- 量化前后的准确率对比
- 内存使用和压缩比分析
- 权重分布分析
- 效率评分计算

## 技术细节

### 优化算法原理
- **梯度下降**: 标准加法更新规则
- **指数梯度**: 乘法更新规则，在log域中操作
- **混合方法**: 结合不同更新策略的优势

### 量化技术
- **后训练量化(PTQ)**: 训练后应用量化
- **量化感知训练(QAT)**: 训练过程中考虑量化影响
- **动态范围**: 自适应缩放和范围裁剪

## 贡献

这是一个学术研究项目，主要贡献包括：
1. 多种优化算法的实现和比较
2. 高效量化方法的开发
3. 综合性能评估框架
4. 详细的实验分析和可视化

## 许可证

本项目仅用于学术研究目的。

## 联系方式

如有问题或建议，请联系项目作者。

---

**注意**: 本项目包含大量实验数据和模型检查点，这些文件通常较大且不适合直接上传到Git。建议使用`.gitignore`文件排除这些内容，或者使用Git LFS进行大文件管理。


