# 【LLMs 入门实战】Ziya-LLaMA-13B-v1 模型学习与实战
 
- Github 代码：https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1

## 一、前言

### 1.1 介绍

该项目开源了姜子牙通用大模型V1，是基于LLaMa的130亿参数的大规模预训练模型，具备翻译，编程，文本分类，信息抽取，摘要，文案生成，常识问答和数学计算等能力。该模型已完成大规模预训练、多任务有监督微调和人类反馈学习三阶段的训练过程。

### 1.2 软件资源

- CUDA 11.7
- Python 3.10
- pytorch 1.13.1+cu117

## 二、环境搭建

### 2.1 下载代码 

### 2.2 构建环境

```s
    $ conda create -n py310_chat python=3.10       # 创建新环境
    $ source activate py310_chat                   # 激活环境
```

### 2.3 安装依赖 

```s
    $ cd Ziya_LLaMA_13B
    $ pip install torch==1.12.1 tokenizers==0.13.3 git+https://github.com/huggingface/transformers
```

### 2.4 下载模型

### 2.5 模型合并

1. Meta 官方发布的 [LLaMA](https://github.com/facebookresearch/llama) 未开源权重，为了遵守相关许可，本项目获取LLaMA权重并转成Hugging Face Transformers模型格式，可参考转换脚本（若已经有huggingface权重则跳过）

```s
  python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 13B --output_dir /output/path
```

2. 下载Ziya-LLaMA-13B-v1的delta权重以及step 1中转换好的原始LLaMA权重，使用如下脚本转换：https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/fengshen/utils/apply_delta.py

```s
  python -m apply_delta --base ~/model_weights/llama-13b --target ~/model_weights/Ziya-LLaMA-13B --delta ~/model_weights/Ziya-LLaMA-13B-v1
```

## 三、模型推理

加载 Ziya-LLaMA-13B-v1 模型推理

```s
  from transformers import AutoTokenizer
  from transformers import LlamaForCausalLM
  import torch

  device = torch.device("cuda")
  ckpt = 'model_weights/Ziya-LLaMA-13B-v1' # '基于delta参数合并后的完整模型权重'

  query="帮我写一份去西安的旅游计划"
  model = LlamaForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float16, device_map="auto")
  tokenizer = AutoTokenizer.from_pretrained(ckpt, use_fast=False)
  inputs = '<human>:' + query.strip() + '\n<bot>:'
        
  input_ids = tokenizer(inputs, return_tensors="pt").input_ids.to(device)
  generate_ids = model.generate(
              input_ids,
              max_new_tokens=1024, 
              do_sample = True, 
              top_p = 0.85, 
              temperature = 1.0, 
              repetition_penalty=1., 
              eos_token_id=2, 
              bos_token_id=1, 
              pad_token_id=0)
  output = tokenizer.batch_decode(generate_ids)[0]
  print(output)
```

## 四、模型微调


## 填坑笔记

## 参考

1. [Ziya-LLaMA-13B-v1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1)
