# MSc Project: Optimization Algorithms and Quantization Methods for Neural Networks

## Project Overview

This project implements and studies various optimization algorithms and quantization methods to improve the efficiency of neural network training and model deployment performance. The project focuses on gradient descent variants, exponential gradient methods, and various quantization techniques (INT4, INT8, LNS) implementation and comparison.

## Key Features

### 🚀 Optimization Algorithms
- **Custom SGD**: Enhanced SGD optimizer supporting both standard gradient descent (GD) and exponential gradient (EG)
- **AdamWeg**: Hybrid optimizer combining Adam and exponential gradient
- **AdamGD**: Hybrid optimizer combining Adam and gradient descent  
- **LNS_Madam**: Madam optimizer based on logarithmic number system

### 🔧 Quantization Methods
- **INT4 Quantization**: Symmetric and asymmetric 4-bit integer quantization
- **INT8 Quantization**: 8-bit integer quantization
- **LNS Quantization**: 4-bit and 8-bit logarithmic number system quantization
- **Mixed Precision**: Support for different precision mixed quantization strategies

### 📊 Model Architectures
- **CNN Model**: Convolutional neural network for image classification
- **MLP Model**: Multi-layer perceptron model
- **Datasets**: MNIST and Fashion-MNIST

## Project Structure

```
MScProject/
├── custom_optim.py          # Custom optimizer implementations
├── lns_quantizer_optimized.py  # LNS quantizer
├── true_int4_quantizer.py   # INT4 quantizer
├── train.py                 # MNIST training script
├── train_fashionmnist.py    # Fashion-MNIST training script
├── CnnModel.py             # Model architecture definitions
├── Data.py                 # Data processing module
├── tests/                  # Unit tests
├── plotscode/              # Plotting and analysis code
└── README.md               # Project documentation
```

## Installation and Usage

### Requirements
- Python 3.8+
- PyTorch 1.9+
- NumPy, Matplotlib, Seaborn
- CUDA support (optional, for GPU acceleration)

### Install Dependencies
```bash
pip install torch torchvision numpy matplotlib seaborn pandas
```

### Run Training
```bash
# MNIST dataset training
python train.py --lr 0.01 --weight_decay 0.001

# Fashion-MNIST dataset training  
python train_fashionmnist.py --lr 0.01 --weight_decay 0.001
```

### Run Quantization
```bash
# INT4 quantization
python quantize_true_int4.py

# LNS quantization
python quantize_lns_optimized.py
```

## Experimental Results

The project includes detailed experimental result analysis, including:
- Training curve comparison of different optimizers
- Accuracy comparison before and after quantization
- Memory usage and compression ratio analysis
- Weight distribution analysis
- Efficiency score calculation

## Technical Details

### Optimization Algorithm Principles
- **Gradient Descent**: Standard additive update rule
- **Exponential Gradient**: Multiplicative update rule, operating in log domain
- **Hybrid Methods**: Combining advantages of different update strategies

### Quantization Techniques
- **Post-Training Quantization (PTQ)**: Applying quantization after training
- **Quantization-Aware Training (QAT)**: Considering quantization impact during training
- **Dynamic Range**: Adaptive scaling and range clipping

## Contributions

This is an academic research project with main contributions including:
1. Implementation and comparison of multiple optimization algorithms
2. Development of efficient quantization methods
3. Comprehensive performance evaluation framework
4. Detailed experimental analysis and visualization

## License

This project is for academic research purposes only.

## Contact

For questions or suggestions, please contact the project author.

---

**Note**: This project contains a large amount of experimental data and model checkpoints, which are typically large and not suitable for direct upload to Git. It's recommended to use the `.gitignore` file to exclude these contents, or use Git LFS for large file management.


