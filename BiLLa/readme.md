# 【LLMs 入门实战】BiLLa 模型学习与实战
 
- 论文名称：BiLLa: A Bilingual LLaMA with Enhanced Reasoning Ability
- Github 代码：https://github.com/Neutralzz/BiLLa

## 一、前言

### 1.1 介绍

BiLLa是开源的推理能力增强的中英双语LLaMA模型。模型的主要特性有：

- 较大提升LLaMA的中文理解能力，并尽可能减少对原始LLaMA英文能力的损伤；
- 训练过程增加较多的任务型数据，利用ChatGPT生成解析，强化模型理解任务求解逻辑；
- 全量参数更新，追求更好的生成效果。

### 1.2 软件资源

- CUDA 11.7
- Python 3.10
- pytorch 1.13.1+cu117

### 1.3 模型简介

该模型以原始LLaMa模型为基础，进行了如下三个阶段的训练。

- 第一阶段：扩充中文词表，使用中文预训练语料Wudao、英文预训练语料PILE、翻译语料WMT的中英数据进行二次预训练。
- 第二阶段：训练数据在第一阶段基础上增加任务型数据，训练过程中两部分数据保持1:1的比例混合。任务型数据均为NLP各任务的主流开源数据，包含有数学解题、阅读理解、开放域问答、摘要、代码生成等，利用ChatGPT API为数据标签生成解析，用于训练提升模型对任务求解逻辑的理解。
- 第三阶段：保留第二阶段任务型数据，并转化为对话格式，增加其他指令数据（如Dolly 2.0、Alpaca GPT4、COIG等），进行对齐阶段的微调。

## 二、环境搭建

### 2.1 下载代码 

```s
  $ git clone https://github.com/Neutralzz/BiLLa
```

### 2.2 构建环境

```s
    $ conda create -n py310_chat python=3.10       # 创建新环境
    $ source activate py310_chat                   # 激活环境
```

### 2.3 安装依赖 

```s
    $ cd BiLLa
    $ pip install torch==1.12.1 tokenizers==0.13.3 git+https://github.com/huggingface/transformers
```

### 2.4 LLaMA 模型 下载并转换

Meta 官方发布的 [LLaMA](https://github.com/facebookresearch/llama) 未开源权重，为了遵守相关许可，本项目获取LLaMA权重并转成Hugging Face Transformers模型格式，可参考转换脚本（若已经有huggingface权重则跳过）

```s
  python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 13B --output_dir /output/path
```

### 2.5 模型下载与使用

本项目开源的模型包含：

- 第二阶段训练的语言模型 [BiLLa-7B-LLM](https://huggingface.co/Neutralzz/BiLLa-7B-LLM)
- 第三阶段指令微调后的模型 [BiLLa-7B-SFT](https://huggingface.co/Neutralzz/BiLLa-7B-SFT)

> 注意：因为LLaMA的License限制，本项目开放的模型权重并不能直接使用。开放的模型权重中word embedding的权重为训练后模型的权重和原始LLaMA权重的和，从而保证拥有LLaMA原始模型授权的开发者可以将本项目发布的模型转化成可以使用的格式。

拥有LLaMA原始模型的开发者可以通过embedding_convert.py完成BiLLa模型权重的还原，以下为示例：

```s
  python embedding_convert.py \
      --model_dir /path_to_BiLLa/BiLLa-7B-SFT \
      --meta_llama_pth_file /path_to_LLaMA/llama-7b/consolidated.00.pth
```


## 三、模型推理

BiLLa-7B-SFT模型的使用可参考eval_codes/get_model_answer.py，下面运行示例是获取该模型的生成结果（用于GPT4打分）：

```s
  python3 get_model_answer.py \
      --model-path /path_to_BiLLa/BiLLa-7B-SFT \
      --model-id billa \
      --question-file table/question_en.jsonl \
      --answer-file table/answer/answer_en_billa.jsonl
```

BiLLa-7B-SFT的模型输入可利用eval_codes/conversation.py的conv_billa构造，也可按以下格式自行构造（注意Assistant:后必须有一个空格）：

```s
  Human: [Your question]
  Assistant: 
```

## 四、模型微调

模型微调 可以 参考 ：https://github.com/Neutralzz/BiLLa/tree/main/train_codes


## 填坑笔记

## 参考

1. [Neutralzz/BiLLa](https://github.com/Neutralzz/BiLLa)
