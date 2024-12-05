# nanoGPT：从0到1构建中文医疗领域小型语言模型

## 项目简介

本项目基于Andrej Karpathy的nanoGPT框架，展示了如何从零开始训练一个专注于医疗领域的小型中文语言模型。项目旨在帮助开发者理解大语言模型的训练流程，并提供一个可复现的医疗领域语言模型实践案例。

## 项目特点

- 使用轻量级nanoGPT框架
- 基于中文医疗问答数据集
- 全流程开源：数据处理、训练、推理

## 项目仓库

- 原始nanoGPT：[https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- 中文数据处理：[https://github.com/ecsfu/nanoGPT_Chinese](https://github.com/ecsfu/nanoGPT_Chinese)

## 数据集

### 数据来源：cMedQA2

- 约10万医学相关问题
- 约20万对应回答
- 专注于医疗领域语料

## 训练详情

### 硬件环境

- GPU：NVIDIA RTX3080 (12GB)

### 训练过程

- 总训练步数：2000步

## 使用示例

1. 下载 `ckpt-7000.pt` 到 `out` 文件夹
2. 运行 `sample.py`
3. 在 `start` 变量中输入提示词

### 生成示例

输入提示：`脸上有痘痘时`

## 结果分析

- 成功生成与提示相关的医疗文本
- 模型能够基于上下文进行文本续写

## 注意事项

⚠️ 模型生成仅供参考，不可作为专业医疗建议。如身体出现不适，请及时就医。

## 局限性与改进方向

1. 生成文本可能包含无关信息
2. 需优化文本生成的停止策略
3. 持续改进模型的上下文相关性

## 快速开始

```bash

# 安装依赖
pip install -r requirements.txt

# 训练模型
python train.py

# 文本生成
python sample.py
```

## 文件说明

- `model.py`：模型架构定义，包含训练和推理实现
- `train.py`：模型训练脚本
- `sample.py`：模型推理和文本生成脚本

## 数据预处理

- 提取问答对并合并到 `input.txt`
- 拆分训练集和验证集
- 使用GPT-2编码
- 生成 `train.bin` 和 `val.bin`
- 详细处理见 `data/medical/prepare.py`

## 贡献与许可

欢迎提交Issues和Pull Requests！

---

*本项目仅用于学习和研究目的*
