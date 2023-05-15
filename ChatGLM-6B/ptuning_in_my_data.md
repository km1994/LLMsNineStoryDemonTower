# 【关于 ChatGLM 特定任务微调】那些你不知道的事

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

本教程主要介绍对于 ChatGLM-6B 模型基于 [P-Tuning v2](https://github.com/THUDM/P-tuning-v2) 的特定任务微调实验，微调目标为自动生成的整数/小数加减乘除运算。

本节 以 整数/小数加减乘除运 数据集为例介绍代码的使用方法，以[yongzhuo/chatglm-maths](https://github.com/yongzhuo/chatglm-maths) 为例。

硬件需求

| **量化等级**   | **最低 GPU 显存**（推理） | **最低 GPU 显存**（高效参数微调） |
| -------------- | ------------------------- | --------------------------------- |
| FP16（无量化） | 13 GB                     | 14 GB                             |
| INT8           | 8 GB                     | 9 GB                             |
| INT4           | 6 GB                      | 7 GB                              |

## 二、环境搭建

### 2.1 构建环境

```s
    $ conda create -n py310_chat python=3.10       # 创建新环境
    $ source activate py310_chat                   # 激活环境
```

### 2.2 下载代码

```s
    $ git clone https://github.com/yongzhuo/chatglm-maths.git
    $ cd chatglm-maths
```

### 2.3 安装依赖

运行微调需要4.27.1版本的transformers。除 ChatGLM-6B 的依赖之外，还需要按照以下依赖

```s
    $ pip install -r requirements.txt
```

## 三、使用方法

### 3.1 训练数据下载

#### 3.1.1 数据来源

1. [tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)
2. [LianjiaTech/BELLE](https://github.com/LianjiaTech/BELLE)
3. [carbonz0/alpaca-chinese-dataset](https://github.com/carbonz0/alpaca-chinese-dataset)

#### 3.1.2 数据介绍

本章使用 [BelleGroup/school_math_0.25M](https://huggingface.co/datasets/BelleGroup/school_math_0.25M) 作为本次特定任务微调实验数据， 该数据集 包含约25万条由BELLE项目生成的中文数学题数据，包含解题过程。

注意：此数据集是由ChatGPT产生的，未经过严格校验，题目或解题过程可能包含错误。使用过程中请注意这一点。

> 样例
```s
{
  "instruction": "题目：小华手里有一个装满糖果的袋子，共有12个，小明想知道里面有多少颗糖果，于是他问小华：“你手里的糖果袋子里有偶数个糖果吗？”小华回答：“有，而且多于10颗。”请问小华手里的糖果袋子里最少有几颗糖果？",
  "input": "",
  "output": "\n由题目可知：小华手里的糖果袋子里有偶数个糖果；\n又知道小华手里的糖果袋子里有多于10颗糖果。\n因为糖果数为偶数，多于10颗，所以糖果数最小必须是12。\n所以小华手里的糖果袋子里最少有12颗糖果。"
}
```

- 字段
  - instruction: 指令
  - input: 输入（本数据集均为空）
  - output: 输出

#### 3.1.3 数据下载方式

> 方式一
```s
from datasets import load_dataset
dataset = load_dataset("BelleGroup/school_math_0.25M")
```

> 方式二【本文所采用的方式】
```s
git lfs install
git clone https://huggingface.co/datasets/BelleGroup/school_math_0.25M
```

### 3.2 模型下载

[Huggingface 平台](https://huggingface.co/THUDM)下载

```s
    $ git lfs install
    $ git clone https://huggingface.co/THUDM/chatglm-6b
```

### 3.3 模型微调

运行以下指令进行微调：

> lora 方式 微调
```s
    $ python c00_toy_lora_train_6b.py
    >>>
    generator_calculate_line: ('13+75=', '13+75=88')
    tokenizer.vocab_size: 150344
    Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:10<00:00,  1.31s/it]
    transformer.word_embeddings.weight False
    ......
    transformer.layers.26.mlp.dense_4h_to_h.bias False
    transformer.layers.27.input_layernorm.weight True
    transformer.layers.27.input_layernorm.bias True
    transformer.layers.27.attention.query_key_value.weight True
    transformer.layers.27.attention.query_key_value.bias True
    transformer.layers.27.attention.dense.weight True
    transformer.layers.27.attention.dense.bias True
    transformer.layers.27.post_attention_layernorm.weight True
    transformer.layers.27.post_attention_layernorm.bias True
    transformer.layers.27.mlp.dense_h_to_4h.weight True
    transformer.layers.27.mlp.dense_h_to_4h.bias True
    transformer.layers.27.mlp.dense_4h_to_h.weight True
    transformer.layers.27.mlp.dense_4h_to_h.bias True
    transformer.final_layernorm.weight True
    transformer.final_layernorm.bias True
    model.chat start
    13+75=88, but that's not the correct answer. The correct answer is 13+75=88, which is 90.
    /anaconda3/envs/py371/lib/python3.7/site-packages/transformers/optimization.py:395: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
    FutureWarning,   
    epoch:   0%|                                 | 0/21 [00:00<?, ?it/s]epochs:
     batch_query: ['简便运算: 98+83= 剖析: 98+83=181']       | 0/8 [00:00<?, ?it/s]
    epoch:   0%|                        | 0/21 [00:00<?, ?it/s]
    epochs:   batch_query: ['简便运算: 98+83= 剖析: 98+83=181']    | 0/8 [00:00<?, ?it/s]
    epoch_global: 0, step_global: 1, step: 0, loss: 4.0625
    batch_query: ['口算: 57.84+13.64 解: 57.84+13.64=71.48']                               epoch_global: 0, step_global: 2, step: 1, loss: 2.5625███▌                 | 2/8 [00:17<00:51,  8.54s/it]
batch_query: ['计算题: 48+1 解答: 48+1=49']                                                                                            epoch_global: 0, step_global: 3, step: 2, loss: 4.15625█████████████████████▎        | 3/8 [00:38<01:09, 13.94s/it]
    batch_query: ['计算题: 61.65+33.05 解答: 61.65+33.05=94.7']                           epoch_global: 0, step_global: 4, step: 3, loss: 2.40625████████████████████████████████████████     | 4/8 [01:01<01:09, 17.43s/it]
batch_query: ['计算: 81+75 回答: 81+75=156']
    ...     
```

### 3.4 模型推理

运行以下指令进行推理：

> lora 方式 推理
```s
    $ python p00_toy_lora_predict_6b.py
    >>>
    generator_calculate_line: ('13+75=', '13+75=88')
    tokenizer.vocab_size: 150344
    eval:   0%|                                                                                                                                                                      | 0/1 [00:00<?, ?it/s]batch_query: ['简便运算: 98+83= 剖析: 98+83=181']
    batch_qtext_0: 简便运算: 98+83= 剖析:
    batch_qans_0: 98+83=181
    response_0: 98+83=171
    {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0, 'bleu': 0.0}
    请输入:
    25.31+86.35=
    请稍等...
    25.31+86.35=101.66
    ...
```

## 四、经验分享

这里主要来着 [yongzhuo/chatglm-maths](https://github.com/yongzhuo/chatglm-maths) 博主在复现过程中 的 经验分享。

1. eps=1e-5(不要改小), 半精度float16, 以及LN采用的是Post-LN(泛化性更好) + DeepNorm, 【害, Attention前也有LN】目的是大模型为了防止梯度溢出等;
2. 模型输入输出, 默认的tokenization_chatglm.py/modeling_chatglm.py不能用, 因为那是完全为生成generate设置的, 需要自己写好所有缩入参数, 或者机子改成适配的;
   1. ChatGLMModel中, get_masks()正常, get_position_ids()函数中‘context_length = seq.index(150004) + 1’ 改为 ‘context_length = len(seq)’;
   2. 训练输入input_ids格式暂定为(训练后post-padding, 推理前pre-padding[tokenization_chatglm.py默认pre-padding]) （eg: x: prompt_1 + "\n" + "_" + text_1 + "\n" + prompt_2 + [gMASK] + [BOS] + "_" + text_2 + [PAD]*N）
   3. 训练输入label_ids格式暂定为(CrossEntropyLoss默认忽略-100不参与计算loss)  (eg：y = [-100]*len(text_1+1) + [BOS] + text_2 + [EOS] + [-100]*N)
   4. 注意position/mask(自带的只是推理用的batch_size=1, 所以训练输入还得自己写), 可参考GLM-130的README.md, huozhe 查看GLM-1源码https://github.com/THUDM/GLM/blob/main/tasks/seq2seq/dataset.py
3. 注意chatglm-6b权重是float16的, 不过计算loss时候会转成float32计算, 最后loss再转回float16更新梯度;
4. ChatGLMTokenizer有时候会报奇奇怪怪的错误, 建议生成时候设置max_new_tokens, 最大{"max_new_tokens": 2048}; decode有时候会出现不存在id;
5. 低秩自适应LORA, RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
   尝试 transformers升级到最新, get_peft_model后再.cuda(), device_map={'':torch.cuda.current_device()}

## 参考/感谢

1. [THUDM/ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)
2. [国产开源类ChatGPT模型，ChatGLM-6b初步微调实验](https://zhuanlan.zhihu.com/p/616013638)
3. [yongzhuo/chatglm-maths](https://github.com/yongzhuo/chatglm-maths) 
4. [THUDM/GLM](https://github.com/THUDM/GLM)
5. [tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)
6. [LianjiaTech/BELLE](https://github.com/LianjiaTech/BELLE)
7. [carbonz0/alpaca-chinese-dataset](https://github.com/carbonz0/alpaca-chinese-dataset)
8. [huggingface/peft](https://github.com/huggingface/peft)
9. [mymusise/ChatGLM-Tuning](https://github.com/mymusise/ChatGLM-Tuning)
10. [国产开源类ChatGPT模型，ChatGLM-6b初步微调实验](https://zhuanlan.zhihu.com/p/616013638) 
