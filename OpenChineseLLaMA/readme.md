# 【LLMs 入门实战】 OpenChineseLLaMA 模型学习与实战
 
- Github 代码：https://github.com/OpenLMLab/OpenChineseLLaMA
- 模型：https://huggingface.co/openlmlab/open-chinese-llama-7b-patch

## 一、前言

### 1.1 介绍

基于 LLaMA-7B 经过中文数据集增量预训练产生的中文大语言模型基座，对比原版 LLaMA，该模型在中文理解能力和生成能力方面均获得较大提升，在众多下游任务中均取得了突出的成绩。

### 1.2 软件资源

- CUDA 11.7
- Python 3.10
- pytorch 1.13.1+cu117

## 二、环境搭建

### 2.1 下载代码 

```s
    $ git clone https://github.com/OpenLMLab/OpenChineseLLaMA
```

### 2.2 构建环境

```s
    $ conda create -n py310_chat python=3.10       # 创建新环境
    $ source activate py310_chat                   # 激活环境
```

### 2.3 安装依赖 

```s
    $ cd OpenChineseLLaMA
    $ pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2.4 下载模型

- 模型名称：Open-Chinese-LLaMA-7B-Patch
- 权重类型：Patch
- 下载地址：
  - [🤗Huggingface]：https://huggingface.co/openlmlab/open-chinese-llama-7b-patch
  - [百度网盘]：https://pan.baidu.com/s/14E7iZKcH-5SHMDu97k70cg?pwd=gk34#list/path=%2F
  - [Google Driver]：https://drive.google.com/drive/folders/1THvuFzq_wojVfMLYV1qsSE_ddSjG0Ypv?usp=sharing
- SHA256：https://github.com/OpenLMLab/OpenChineseLLaMA/blob/main/SHA256.txt

### 2.5 模型合并

Meta 官方发布的 [LLaMA](https://github.com/facebookresearch/llama) 未开源权重，为了遵守相关许可，本次发布的模型为 补丁（Patch） 类型，须配合原始官方权重才可以使用。

官方 提供了 补丁（Patch） 的安装脚本，在通过正规渠道获得官方权重后，可以通过以下方式安装补丁：

```s
  python tools/patch_model.py --base_model <path_or_name_to_original_model>
                              --patch_model openlmlab/open-chinese-llama-7b-patch
                              --base_model_format <hf_or_raw>
```

> 提示：本补丁的安装方式为原地安装，即安装后的补丁即为完整版 huggingface 版本的本模型权重，您可以使用 transformers加载模型。

## 三、模型推理

可以使用脚本启动交互式界面：

```s
  python cli_demo.py --model openlmlab/open-chinese-llama-7b-patch
                    --devices 0
                    --max_length 1024
                    --do_sample true
                    --top_k 40
                    --top_p 0.8
                    --temperature 0.7
                    --penalty 1.02
```

## 四、模型微调



## 填坑笔记


## 参考

1. [Panda LLM: Training Data and Evaluation for Open-Sourced Chinese Instruction-Following Large Language Models](https://arxiv.org/pdf/2305.03025v1.pdf)
2. [dandelionsllm/pandallm](https://github.com/dandelionsllm/pandallm)
