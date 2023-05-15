# 【LLMs 入门实战 —— 六 】Chinese-LLaMA-Alpaca 模型学习与实战

- github 地址: https://github.com/ydli-ai/Chinese-ChatLLaMA
- 试用地址：https://alpaca-ai-custom6.ngrok.io/

## 【LLMs 入门实战系列】

### 第一层 ChatGLM-6B

1. [【ChatGLM-6B入门-一】清华大学开源中文版ChatGLM-6B模型学习与实战](ChatGLM-6B/induction.md)
   1. 介绍：ChatGLM-6B 环境配置 和 部署
2. [【ChatGLM-6B入门-二】清华大学开源中文版ChatGLM-6B模型微调实战](ChatGLM-6B/ptuning.md)
   1. ChatGLM-6B P-Tuning V2 微调：Fine-tuning the prefix encoder of the model.
3. [【ChatGLM-6B入门-三】ChatGLM 特定任务微调实战](https://articles.zsxq.com/id_3b42ukjdkwpt.html)
4. [【ChatGLM-6B入门-四】ChatGLM + LoRA 进行finetune](https://articles.zsxq.com/id_e2389qm0w0sx.html)
   1. 介绍：ChatGLM-6B LoRA 微调：Fine-tuning the low-rank adapters of the model.
5. [ChatGLM-6B 小编填坑记](https://articles.zsxq.com/id_fw7vn0mhdsnq.html)
   1. 介绍：ChatGLM-6B 在 部署和微调 过程中 会遇到很多坑，小编掉坑了很多次，为防止 后人和小编一样继续掉坑，小编索性把遇到的坑都填了。
6. [【LLMs学习】关于大模型实践的一些总结](https://articles.zsxq.com/id_il58nxrs9jxr.html)
7. [【LLMs 入门实战 —— 十一 】基于 🤗PEFT 的高效 🤖ChatGLM-6B 微调](https://articles.zsxq.com/id_7rz5jtfguuc5.html)
   1. 微调方式：
      1. ChatGLM-6B Freeze 微调：Fine-tuning the MLPs in the last n blocks of the model.
      2. ChatGLM-6B P-Tuning V2 微调：Fine-tuning the prefix encoder of the model.
      3. ChatGLM-6B LoRA 微调：Fine-tuning the low-rank adapters of the model.
8. [【LLMs 入门实战 —— 十二 】基于 本地知识库 的高效 🤖langchain-ChatGLM ](https://articles.zsxq.com/id_54vjwns5t6in.html)
   1. 介绍：langchain-ChatGLM是一个基于本地知识的问答机器人，使用者可以自由配置本地知识，用户问题的答案也是基于本地知识生成的。

### 第二层 Stanford Alpaca 7B 

- [【LLMs 入门实战 —— 五 】Stanford Alpaca 7B 模型学习与实战](https://articles.zsxq.com/id_xnt3fvp2wxz0.html)
  - 介绍：本教程提供了对LLaMA模型进行微调的廉价亲民 LLMs 学习和微调 方式，主要介绍对于 Stanford Alpaca 7B 模型在特定任务上 的 微调实验，所用的数据为OpenAI提供的GPT模型API生成质量较高的指令数据（仅52k）。

### 第三层 Chinese-LLaMA-Alpaca 

- [【LLMs 入门实战 —— 六 】Chinese-LLaMA-Alpaca 模型学习与实战](https://articles.zsxq.com/id_dqvusswrdg6c.html)
  - 介绍：本教程主要介绍了 Chinese-ChatLLaMA,提供中文对话模型 ChatLLama 、中文基础模型 LLaMA-zh 及其训练数据。 模型基于 TencentPretrain 多模态预训练框架构建

### 第四层 小羊驼 Vicuna

- [【LLMs 入门实战 —— 七 】小羊驼 Vicuna模型学习与实战](https://articles.zsxq.com/id_q9mx24q9fdab.html)
  - 介绍：UC伯克利学者联手CMU、斯坦福等，再次推出一个全新模型70亿/130亿参数的Vicuna，俗称「小羊驼」（骆马）。小羊驼号称能达到GPT-4的90%性能

### 第五层 MiniGPT-4 

- [【LLMs 入门实战 —— 八 】MiniGPT-4 模型学习与实战](https://articles.zsxq.com/id_ff0w6czthq25.html)
  - 介绍： MiniGPT-4，是来自阿卜杜拉国王科技大学的几位博士做的，它能提供类似 GPT-4 的图像理解与对话能力

### 第六层 GPT4ALL

- [【LLMs 入门实战 —— 八 】GPT4ALL 模型学习与实战](https://articles.zsxq.com/id_ff0w6czthq25.html)
  - 介绍：一个 可以在自己笔记本上面跑起来的  Nomic AI 的助手式聊天机器人，成为贫民家孩子的 福音！

### 第七层 AutoGPT

- [AutoGPT 使用和部署](https://articles.zsxq.com/id_pli0z9916126.html)
  - 介绍：Auto-GPT是一个基于ChatGPT的工具，他能帮你自动完成各种任务，比如写代码、写报告、做调研等等。使用它时，你只需要告诉他要扮演的角色和要实现的目标，然后他就会利用ChatGPT和谷歌搜索等工具，不断“思考”如何接近目标并执行，你甚至可以看到他的思考过程。

### 第八层 MOSS

- [【LLMs 入门实战 —— 十三 】MOSS 模型学习与实战](https://articles.zsxq.com/id_4vwpxod23zrc.html)
  - 介绍：MOSS是一个支持中英双语和多种插件的开源对话语言模型，moss-moon系列模型具有160亿参数，在FP16精度下可在单张A100/A800或两张3090显卡运行，在INT4/8精度下可在单张3090显卡运行。MOSS基座语言模型在约七千亿中英文以及代码单词上预训练得到，后续经过对话指令微调、插件增强学习和人类偏好训练具备多轮对话能力及使用多种插件的能力。
  - 局限性：由于模型参数量较小和自回归生成范式，MOSS仍然可能生成包含事实性错误的误导性回复或包含偏见/歧视的有害内容，请谨慎鉴别和使用MOSS生成的内容，请勿将MOSS生成的有害内容传播至互联网。若产生不良后果，由传播者自负。



## 一、前言

本教程主要介绍了 [Chinese-ChatLLaMA](https://github.com/ydli-ai/Chinese-ChatLLaMA),提供中文对话模型 ChatLLama 、中文基础模型 LLaMA-zh 及其训练数据。 模型基于 [TencentPretrain](https://github.com/Tencent/TencentPretrain) 多模态预训练框架构建

## 二、整体方法介绍

ChatLLaMA 支持简繁体中文、英文、日文等多语言。 LLaMA 在预训练阶段主要使用英文，为了将其语言能力迁移到中文上，首先进行中文增量预训练， 使用的语料包括中英平行语料、中文维基、社区互动、新闻数据、科学文献等。再通过 Alpaca 指令微调得到 Chinese-ChatLLaMA。

- **项目特点**
  - 通过 Full-tuning （全参数训练）获得中文模型权重，提供 TencentPretrain 与 HuggingFace 版本
  - 模型细节公开可复现，提供数据准备、模型训练和模型评估完整流程代码
  - 提供目前最大的中文 LLaMA 模型
  - 多种量化方案，支持 CUDA 和边缘设备部署推理

## 三、环境搭建

### 3.1 基础环境配置要求

1. 操作系统：Linux
2. CPUs: 单个节点具有 1TB 内存的 Intel CPU，物理CPU个数为64，每颗CPU核数为16
3. GPUs: 8 卡 A800 80GB GPUs
4. Python: 3.10 

### 3.2 构建环境

```s
    $ conda create -n py310_chat python=3.10       # 创建新环境
    $ source activate py310_chat                   # 激活环境
```

### 3.3 离线安装 pytorch

离线安装PyTorch，点击下载对应 [cuda版本](https://download.pytorch.org/whl/torch_stable.html)的torch和torchvision即可。

```s
    $ pip  install -U torch==1.13.1
    $ pip install torchvision-0.14.1
```

### 3.4 安装 TencentPretrain

安装 TencentPretrain ，后面需要用他 转化 预训练模型 

```s
    $ https://github.com/Tencent/TencentPretrain.git
```

### 3.5 安装 apex

```s
    $ git clone https://github.com/NVIDIA/apex.git
    $ cd apex
    $ git checkout 22.04-dev
    $ pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### 3.6 安装其他依赖包

使用 以下命令 安装其他依赖包

```s
    $ pip install -r requirements.txt
```

> 注：**如果到这里一路畅通，那就说明 你已经 完成一大步工作了！**

## 四、LLaMA原始权重文件 下载

```s
    $ git lfs install
    $ git clone https://huggingface.co/P01son/ChatLLaMA-zh-7B
    >>>
    Cloning into 'ChatLLaMA-zh-7B'...
    remote: Enumerating objects: 12, done.
    remote: Counting objects: 100% (12/12), done.
    remote: Compressing objects: 100% (11/11), done.
    remote: Total 12 (delta 2), reused 0 (delta 0), pack-reused 0
    Unpacking objects: 100% (12/12), 1.44 KiB | 369.00 KiB/s, done.
    ...
```

## 五、训练数据介绍

使用的语料包括[中英平行语料](https://statmt.org/wmt18/translation-task.html#download)、[中文维基](https://github.com/CLUEbenchmark/CLUECorpus2020)、[社区互动](https://github.com/CLUEbenchmark/CLUECorpus2020)、[新闻数据](https://github.com/CLUEbenchmark/CLUECorpus2020)、[科学文献](https://github.com/ydli-ai/CSL)等

下载语料后，合并到一个 .txt 文件并按行随机打乱，语料格式如下：

doc1
doc2
doc3

## 六、Chinese-ChatLLaMA 快速使用

### 6.1 clone fengyh3/llama_inference 推理

```s
    $ git clone https://github.com/fengyh3/llama_inference.git
    Cloning into 'llama_inference'...
    remote: Enumerating objects: 145, done.
    remote: Counting objects: 100% (145/145), done.
    remote: Compressing objects: 100% (102/102), done.
    remote: Total 145 (delta 68), reused 105 (delta 35), pack-reused 0
    Receiving objects: 100% (145/145), 47.15 KiB | 41.00 KiB/s, done.
    Resolving deltas: 100% (68/68), done.
```

### 6.2 fengyh3/llama_inference 问答

```s
    $ cd llama_inference 
    $ vi prompts.txt  #编辑用户输入，例如"上海有什么好玩的地方？"
    $ python llama_infer.py --test_path prompts.txt --prediction_path result.txt  --load_model_path /mnt/kaimo/data/chat/ChatLLaMA-zh-7B/chatllama_7b.bin   --config_path config/llama_7b_config.json  --spm_model_path /mnt/kaimo/data/chat/ChatLLaMA-zh-7B/tokenizer.model --seq_length 512
```

- 参数介绍：
  - test_path：测试文件 路径
  - prediction_path：预测结果文件 路径
  - load_model_path：ChatLLaMA-zh-7B 模型 路径
  - config_path：ChatLLaMA-zh-7B 模型 配置项 路径
  - spm_model_path：ChatLLaMA-zh-7B 模型 tokenizer 路径
  - seq_length：序列长度

### 6.3 fengyh3/llama_inference Int8 推理加速

如果 显存不够，可以 使用  Int8 推理加速，只需要在 后面加上 -use_int8  即可

```s
    $ python llama_infer.py --test_path beginning.txt --prediction_path result.txt  \
        --load_model_path ../ChatLLaMA-zh-7B/chatllama_7b.bin  \
        --config_path config/llama_7b_config.json \
        --spm_model_path ../ChatLLaMA-zh-7B/tokenizer.model --seq_length 512 --use_int8 
```

## 七、Chinese-ChatLLaMA 模型训练

### 7.1 clone TencentPretrain 

```s
    $ git clone https://github.com/Tencent/TencentPretrain.git
    $ cd TencentPretrain
```

### 7.2 下载[预训练LLaMA权重](https://huggingface.co/decapoda-research/llama-7b-hf)

```s
    $ git lfs install
    $ git clone https://huggingface.co/decapoda-research/llama-7b-hf
```

### 7.3 [预训练LLaMA权重](https://huggingface.co/decapoda-research/llama-7b-hf)转化为TencentPretrain格式

```s
    $ python scripts/convert_llama_from_huggingface_to_tencentpretrain.py --input_model_path $LLaMA_HF_PATH --output_model_path  models/llama-7b.bin --type 7B
```

> 注：这里的 $LLaMA_HF_PATH 为 [预训练LLaMA权重](https://huggingface.co/decapoda-research/llama-7b-hf) 下载地址

### 7.4 下载[中文预训练语料](https://github.com/ydli-ai/Chinese-ChatLLaMA/blob/main/corpus/README.md)， 预处理

```s
    $ python preprocess.py --corpus_path $CORPUS_PATH --spm_model_path $LLaMA_PATH/tokenizer.model  --dataset_path $OUTPUT_DATASET_PATH --data_processor lm --seq_length 512
```

## 八、总结


## 踩坑手册

### ModuleNotFoundError: No module named 'bitsandbytes'

- 问题描述：

```s
    Traceback (most recent call last):
    File "llama_infer.py", line 4, in <module>
        from model.llama import *
    File "/mnt/kaimo/project/chat_wp/llama_inference/model/llama.py", line 6, in <module>
        import bitsandbytes as bnb
    ModuleNotFoundError: No module named 'bitsandbytes'
```

- 解决方法：

```s
    $ pip install bitsandbytes
```

## 参考

1. [ymcui/Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
2. [facebookresearch/llama](https://github.com/facebookresearch/llama)
3. [stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)
4. [alpaca-lora](https://github.com/tloen/alpaca-lora)
5. [Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html)
6.  [llama-7b-hf/tree/main](https://huggingface.co/decapoda-research/llama-7b-hf/tree/main)
