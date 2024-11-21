# LLM Training From Scratch

这是一个从零开始训练大规模语言模型的开源项目。本项目旨在提供一个清晰、可扩展的代码框架，帮助理解和实现Transformer架构的语言模型训练。

## 项目特点

- 🚀 模块化的代码结构，便于理解和扩展
- 📦 完整的训练流程实现
- 🔧 可配置的模型参数和训练设置
- 📈 训练过程监控和可视化
- 🔄 支持断点续训
- 📊 详细的评估指标

## 安装

1. 克隆项目

```bash
git clone https://github.com/yangjie-ai/llm-training.git
cd llm-training
```

2. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows
```

3. 安装依赖

```bash
pip install -r requirements.txt
```

## 项目结构

```
project_root/
├── config/          # 配置文件
├── src/             # 源代码
├── scripts/         # 训练和评估脚本
├── tests/           # 测试代码
└── outputs/         # 输出目录
```

## 使用方法

1. 数据预处理

```bash
python scripts/preprocess.py --config config/data_config.yaml
```

2. 训练模型

```bash
python scripts/train.py --config config/training_config.yaml
```

3. 评估模型

```bash
python scripts/evaluate.py --model-path outputs/checkpoints/latest
```

## 配置说明

模型和训练的配置都在 `config` 目录下：

- `model_config.yaml`: 模型架构配置
- `training_config.yaml`: 训练超参数配置
- `data_config.yaml`: 数据处理配置

## 开发路线图

- [X]  项目基础结构搭建
- [ ]  数据预处理流程
- [ ]  Transformer模型实现
- [ ]  训练流程搭建
- [ ]  分布式训练支持
- [ ]  模型评估与分析工具

## 贡献指南

欢迎提交Issue和Pull Request！请确保在提交PR之前：

1. 更新测试用例
2. 遵循项目的代码规范
3. 更新相关文档

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。
