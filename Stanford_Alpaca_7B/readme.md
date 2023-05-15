# 【LLMs 入门实战 —— 五 】Stanford Alpaca 7B 模型学习与实战

- github 地址: https://github.com/tatsu-lab/stanford_alpaca
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

目前，LLMs 满天飞，但是想要 训练高质量的指令遵循模型(instruction-following model)面临两个重要挑战：

1. 强大的预训练语言模型；
2. 高质量的指令遵循数据；

对于第一个挑战，最近，Meta开源了他们的LLaMA系列模型，包含了参数量为7B/13B/33B/65B的不同模型，但是Stanford 科学家们发现 原模型的效果较差（如生成的结果文不对题、并且无法自然地结束生成等）。

对于第二个挑战，Self-Instruct 提出一种利用现有的强大语言模型自动生成指令数据。因此，斯坦福的 Alpaca 模型基于 LLaMA-7B模型结合self-instruct 方式生成的52k指令遵循(instruction-following)样本数据进行有监督的指令微调，就能达到类似 GPT-3.5 的效果。

本教程提供了对LLaMA模型进行微调的廉价亲民 LLMs 学习和微调 方式，主要介绍对于 Stanford Alpaca 7B 模型在特定任务上 的 微调实验，所用的数据为OpenAI提供的GPT模型API生成质量较高的指令数据（仅52k）。

## 二、整体方法介绍

1. 利用OpenAI提供的GPT模型(text-davinci-003)API结合self-instruct方法生成质量较高的指令数据（仅52k），例如：

```s
{
    "instruction": "Rewrite the following sentence in the third person",
    "input": "I am anxious",
    "output": "She is anxious."
}, {
    "instruction": "What are the three primary colors?",
    "input": "",
    "output": "The three primary colors are red, blue, and yellow."
}
```

具体操作方法：使用 self-instruct 种子集中的 175 个人工编写的指令-输出(instruction-output)对，然后用该种子集作为 in-context 样本 prompt text-davinci-003模型来生成更多指令。Alpaca通过简化生成 pipeline 改进了 self-instruct 方法，并显著降低了成本。

> Alpaca官方声称基于openai的API生成52k指令数据集的费用<500美元。关于self-instruct 方法的细节留待后续在本系列中详细说明，敬请期待留意。

2. 基于这些指令数据使用HuggingFace Transformers框架精调LLaMA-7B模型。

![](img/微信截图_20230414124431.png)

> 注：在这个过程利用了FSDP(Fully Sharded Data Parallel)和混合精度训练等技术。成本方面，Alpaca在8个80GB A100 上微调一个 7B LLaMA 模型需要3个小时，这对大多数云计算提供商来说成本不到 100 美元。整体价格还算比较亲民，可盐可甜。

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

### 3.4 安装 transformers

安装 transformers ，目前，[LLaMA相关的实现](https://github.com/huggingface/transformers/commit/0041be5b3d1b9a5e1443e1825d7d80f6dfadcdaa)并没有发布对应的版本，但是已经合并到主分支了，因此，我们需要切换到对应的commit，从源代码进行相应的安装。

```s
    $ https://github.com/huggingface/transformers.git
    $ cd transformers
    $ git checkout 0041be5 
    $ pip3 install . -i https://mirrors.cloud.tencent.com/pypi/simple
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

requirements.txt文件具体的内容如下。

```s
    numpy
    rouge_score
    fire
    openai
    sentencepiece
    tokenizers==0.12.1
    wandb
    deepspeed==0.8.0
    accelerate
    tensorboardX
```

> 注：**如果到这里一路畅通，那就说明 你已经 完成一大步工作了！**

## 四、模型格式转换

### 4.1 LLaMA原始权重文件 下载

根据自己的需求下载泄露的LLaMA的7B、13B等等的模型权重，需要填form找facebook申请，但很难得到回复，演示使用的是最小的7B版，

1. 下载方式一： 使用 aria2c 下载

```s
    For the 7B model...
    aria2c --select-file 21-23,25,26 'magnet:?xt=urn:btih:b8287ebfa04f879b048d4d4404108cf3e8014352&dn=LLaMA'
    https://huggingface.co/nyanko7/LLaMA-7B/tree/main

    For the 13B model...

    aria2c --select-file 1-4,25,26 'magnet:?xt=urn:btih:b8287ebfa04f879b048d4d4404108cf3e8014352&dn=LLaMA'
    For the 30B model...

    aria2c --select-file 5-10,25,26 'magnet:?xt=urn:btih:b8287ebfa04f879b048d4d4404108cf3e8014352&dn=LLaMA'
    For the 65B model...

    aria2c --select-file 11-20,25,26 'magnet:?xt=urn:btih:b8287ebfa04f879b048d4d4404108cf3e8014352&dn=LLaMA'

    And for everything...

    aria2c 'magnet:?xt=urn:btih:b8287ebfa04f879b048d4d4404108cf3e8014352&dn=LLaMA' 
```


2. 下载方式二：百度云盘下载，[百度网盘的下载链接](https://pan.baidu.com/s/19g792GUMtELGBMCfu0d2OQ?pwd=x2ck)

3. 下载方式三：通过pyllama下载

- step 1：安装pyllama

```s
    $ pip install pyllama -U
```

- step 2：下载7B的模型

```s
    $ python -m llama.download --model_size 7B
```

- step 3：当然你也可以下载更大的模型，有7B,13B,30B,65B共计4种。

4. 下载方式四：通过ipfs下载

这个应该是最早泄漏的LLaMA模型，地址为 https://ipfs.io/ipfs/Qmb9y5GCkTG7ZzbBWMu2BXwMkzyCKcUjtEKPpgdZ7GEFKm/

- step 1：首先安装ipfs客户端，最好用带界面的。https://docs.ipfs.tech/install/ipfs-desktop/
- step 2：然后7B模型的index为：QmbvdJ7KgvZiyaqHw5QtQxRtUd7pCAdkWWbzuvyKusLGTw

### 4.2 LLaMA原始权重文件格式转换

需要将LLaMA原始权重文件转换为Transformers库对应的模型文件格式。如果不想自己转，也可以直接从Hugging Face下载转换好的模型

```s
   $ cd transformers

    $ python src/transformers/models/llama/convert_llama_weights_to_hf.py \ 
    --input_dir data/llama-model \
    --model_size 7B \
    --output_dir data/hf-llama-model
```

这个版本transformers转换得到的结果是分别存于2个文件夹：llama-7b和tokenizer

可以通过以下方式加载模型和分词器：

```s
    $ tokenizer = transformers.LlamaTokenizer.from_pretrained("data/llama-model/tokenizer/")

    $ model = transformers.LlamaForCausalLM.from_pretrained("data/llama-model/llama-7b/")
```

LLaMA 分词器（tokenizer）基于 [sentencepiece分词工具](https://github.com/google/sentencepiece)。 sentencepiece在解码序列时，如果第一个token是单词（例如：Banana）开头，则tokenizer不会在字符串前添加前缀空格。 要让tokenizer输出前缀空格，请在LlamaTokenizer对象或tokenizer配置中设置decode_with_prefix_space=True。

为了方便将tokenizer目录的文件拷贝到llama-7b目录下。如果是直接用最新版的transformers中转换脚本的话在hf-llama-model会将模型参数文件和tokenizer相关文件平铺放一起。

```s
    $ cp tokenizer/* llama-7b/
```

> 注：注: 如果不想转换也可以直接从Hugging Face下载转换好的 [模型](https://huggingface.co/decapoda-research/llama-7b-hf)。

## 五、数据准备

 [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) 中的alpaca_data.json文件即是他们用于训练的指令数据集，我们可以直接使用该数据集进行模型精调。但是在 [Alpaca-LoRA](https://github.com/tloen/alpaca-lora) 中提到该数据集存在一些噪声，因此，他们对该数据集做了清洗后得到了 [alpaca_data_cleaned_archive.json](https://github.com/tloen/alpaca-lora/blob/main/alpaca_data_cleaned_archive.json) 文件。采用该数据集进行训练大概率会得到更好结果。

## 六、模型精调

运行命令

```s
    torchrun --nproc_per_node=8 --master_port=25001 train.py \
        --model_name_or_path  pretrain/hf-llama-model/llama-7b \
        --data_path data/alpaca_data_cleaned.json \
        --bf16 True \
        --output_dir output/alpaca/sft_7b \
        --num_train_epochs 1 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 2000 \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --report_to "tensorboard" \
        --fsdp "full_shard auto_wrap" \
        --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
        --tf32 True
```

> 微调优化 可以看 踩坑手册->（2）显存占用高和训练效率慢 问题 章节

> 微调过程

```s
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
*****************************************
[2023-03-28 11:13:02,320] [INFO] [comm.py:657:init_distributed] Initializing TorchBackend in DeepSpeed with backend nccl
[2023-03-28 11:13:20,236] [INFO] [partition_parameters.py:413:__exit__] finished initializing model with 6.74B parameters
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 33/33 [00:41<00:00,  1.26s/it]
...
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 33/33 [00:41<00:00,  1.26s/it]
Using pad_token, but it is not set yet.
...
Using pad_token, but it is not set yet.
WARNING:root:Loading data...
...
WARNING:root:Loading data...
WARNING:root:Formatting inputs...
...
WARNING:root:Formatting inputs...
WARNING:root:Tokenizing inputs... This may take some time...
..
WARNING:root:Tokenizing inputs... This may take some time...
Using /base/.cache/torch_extensions/py310_cu113 as PyTorch extensions root...
...
Using /base/.cache/torch_extensions/py310_cu113 as PyTorch extensions root...
Emitting ninja build file /base/.cache/torch_extensions/py310_cu113/utils/build.ninja...
Building extension module utils...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
Using /base/.cache/torch_extensions/py310_cu113 as PyTorch extensions root...
ninja: no work to do.
Loading extension module utils...
...
Loading extension module utils...
Time to load utils op: 0.10286140441894531 seconds
...
Time to load utils op: 0.20401406288146973 seconds
Parameter Offload: Total persistent parameters: 0 in 0 params
Using /base/.cache/torch_extensions/py310_cu113 as PyTorch extensions root...
No modifications detected for re-loaded extension module utils, skipping build step...
Using /base/.cache/torch_extensions/py310_cu113 as PyTorch extensions root...Loading extension module utils...

Time to load utils op: 0.0004200935363769531 seconds
No modifications detected for re-loaded extension module utils, skipping build step...
Loading extension module utils...
Using /base/.cache/torch_extensions/py310_cu113 as PyTorch extensions root...
Time to load utils op: 0.0003352165222167969 seconds
No modifications detected for re-loaded extension module utils, skipping build step...
Loading extension module utils...
Time to load utils op: 0.0003571510314941406 seconds
Using /base/.cache/torch_extensions/py310_cu113 as PyTorch extensions root...
Using /base/.cache/torch_extensions/py310_cu113 as PyTorch extensions root...Using /base/.cache/torch_extensions/py310_cu113 as PyTorch extensions root...

No modifications detected for re-loaded extension module utils, skipping build step...
Loading extension module utils...
No modifications detected for re-loaded extension module utils, skipping build step...
Loading extension module utils...
No modifications detected for re-loaded extension module utils, skipping build step...
Loading extension module utils...
Time to load utils op: 0.0006623268127441406 seconds
Time to load utils op: 0.0005290508270263672 seconds
Time to load utils op: 0.0006077289581298828 seconds
Using /base/.cache/torch_extensions/py310_cu113 as PyTorch extensions root...
No modifications detected for re-loaded extension module utils, skipping build step...
Loading extension module utils...
Time to load utils op: 0.001024484634399414 seconds
Using /base/.cache/torch_extensions/py310_cu113 as PyTorch extensions root...
No modifications detected for re-loaded extension module utils, skipping build step...
Loading extension module utils...
Time to load utils op: 0.0003275871276855469 seconds
{'loss': 1.5163, 'learning_rate': 0.0, 'epoch': 0.01}
{'loss': 1.5216, 'learning_rate': 0.0, 'epoch': 0.02}
...
{'loss': 1.0547, 'learning_rate': 2.025571894372794e-06, 'epoch': 0.98}
{'loss': 1.0329, 'learning_rate': 1.8343633694278895e-06, 'epoch': 0.99}
{'loss': 1.0613, 'learning_rate': 1.6517194697072903e-06, 'epoch': 1.0}
{'train_runtime': 4605.8781, 'train_samples_per_second': 11.277, 'train_steps_per_second': 0.022, 'train_loss': 1.175760779050317, 'epoch': 1.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [1:16:45<00:00, 45.60s/it]
...
```

> GPU显存占用情况

```s
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.161.03   Driver Version: 470.161.03   CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A800 80G...  Off  | 00000000:34:00.0 Off |                    0 |
| N/A   47C    P0    75W / 300W |  66615MiB / 80994MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA A800 80G...  Off  | 00000000:35:00.0 Off |                    0 |
| N/A   46C    P0    70W / 300W |  31675MiB / 80994MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   2  NVIDIA A800 80G...  Off  | 00000000:36:00.0 Off |                    0 |
| N/A   49C    P0    72W / 300W |  35529MiB / 80994MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   3  NVIDIA A800 80G...  Off  | 00000000:37:00.0 Off |                    0 |
| N/A   50C    P0    76W / 300W |  54277MiB / 80994MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   4  NVIDIA A800 80G...  Off  | 00000000:9B:00.0 Off |                    0 |
| N/A   51C    P0    80W / 300W |  44229MiB / 80994MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   5  NVIDIA A800 80G...  Off  | 00000000:9C:00.0 Off |                    0 |
| N/A   49C    P0    72W / 300W |  59841MiB / 80994MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   6  NVIDIA A800 80G...  Off  | 00000000:9D:00.0 Off |                    0 |
| N/A   47C    P0    77W / 300W |  65217MiB / 80994MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   7  NVIDIA A800 80G...  Off  | 00000000:9E:00.0 Off |                    0 |
| N/A   43C    P0    68W / 300W |  30141MiB / 80994MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     30523      C   ...py310/bin/python    63693MiB |
|    1   N/A  N/A     30235      C   ...py310/bin/python    31653MiB |
|    2   N/A  N/A     30136      C   ...py310/bin/python    35507MiB |
|    3   N/A  N/A     31537      C   ...py310/bin/python    53655MiB |
|    4   N/A  N/A     30540      C   ...py310/bin/python    44207MiB |
|    5   N/A  N/A     30542      C   ...py310/bin/python    59619MiB |
|    6   N/A  N/A     30522      C   ...py310/bin/python    65195MiB |
|    7   N/A  N/A     30743      C   ...py310/bin/python    30119MiB |
+-----------------------------------------------------------------------------+
```


## 七、模型推理

> 实例一
```s
### Instruction:你是一名小学生，谈谈你的理想。
### Response:
我的理想是想成为一名科学家。从事有关科学研究，努力推动科学研究发展，同时还能给人们带来实质性的改变和帮助。我希望能够在有关的研究领域取得杰出的成就，做出对人类社会的贡献。
```

> 实例二
```s
### Instruction:你是一名大学生，说一下你的理想。
### Response:
我的理想是成为一名优秀的教师，能给学生们提供更多的机会，帮助他们成长和发展，做到他们的爱好和梦想。我希望能够让学生们充分理解课程的内容，深入探索，充分发挥自己的潜能，最终取得成功。我还希望能够与家长和其他监督人有更好的沟通，让学生们在学习和生活中拥有更多的 healthy competition，最终达到学习的目的。
```

## 八、总结

由于 LLaMA 在预训练阶段所用的语料基本都是英文语料，所以 LLaMA 在英文上具有强大的零样本学习和迁移能力，但是在 中文能力上 相比于 清华大学 的 CHatGLM 偏很弱。

## 踩坑手册

### （1） Cuda 版本太低问题

Stanford Alpaca 7B 需要的 Cuda版本 为 11.6及以上，且PyTorch版本升级为1.13.1及以上

### （2）显存占用高和训练效率慢 问题

- 动机：因为 Stanford Alpaca 7B 微调 显存占用高和训练效率慢
- 解决方法：使用DeepSpeed框架来减少显存占用和提高训练效率
- 具体操作

1. clone stanford_alpaca 项目

```s
git clone https://github.com/tatsu-lab/stanford_alpaca.git
cd stanford_alpaca
```

2. 修改train.py文件

```s
    # 注释掉原有代码
    """
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    """
    # 通过Llama加载tokenizer和model
    model = transformers.LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    
    trainer.save_state()
    # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    trainer.save_model()
```

3. 启动命令

```s
    torchrun --nproc_per_node=8 --master_port=11223 train.py \
    --model_name_or_path pretrain/hf-llama-model/llama-7b \
    --data_path data/alpaca_data_cleaned.json \
    --output_dir output/alpaca/sft_7b \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --fp16 True \
    --deepspeed ds_config.json
```

> 其中，ds_config.json文件内容如下所示

```s
{
    "zero_optimization": {
        "stage": 3,
        "contiguous_gradients": true,
        "stage3_max_live_parameters": 0,
        "stage3_max_reuse_distance": 0,
        "stage3_prefetch_bucket_size": 0,
        "stage3_param_persistence_threshold": 1e2,
        "reduce_bucket_size": 1e2,
        "sub_group_size": 1e8,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "fp16": {
        "enabled": true,
        "auto_cast": false,
        "loss_scale": 0,
        "initial_scale_power": 32,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

### （3）运行完最后一个epoch后出现OOM

- 问题描述：运行完最后一个epoch后出现OOM

```s
    {'train_runtime': 5162.7837, 'train_samples_per_second': 10.072, 'train_steps_per_second': 0.157, 'train_loss': 1.0267484738615347, 'epoch': 1.0}
    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 812/812 [1:26:02<00:00,  6.36s/it]

    /opt/python3.10.11/lib/python3.10/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py:2224: 
    UserWarning: Failed to clone() tensor with name _fsdp_wrapped_module._fpw_module.model.layers.28.mlp.down_proj.weight. This may mean that this state_dict entry could point to invalid memory regions after returning from 
    state_dict() call if this parameter is managed by FSDP. 
    Please check clone implementation of _fsdp_wrapped_module._fpw_module.model.layers.28.mlp.down_proj.weight. 
    Error: CUDA out of memory. Tried to allocate 172.00 MiB (GPU 3; 39.59 GiB total capacity; 35.81 GiB already allocated; 
    79.19 MiB free; 37.59 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to 
    avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```

- 解决方法：参考官方Repo上issue，将safe_save_model_for_hf_trainer改为如下：

```s
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    # state_dict = trainer.model.state_dict()
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        BackwardPrefetch,
        ShardingStrategy,
        FullStateDictConfig,
        StateDictType,
    )
    model=trainer.model  
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state_dict = model.state_dict()
    if trainer.args.should_save:
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
```

## 参考

1. [facebookresearch/llama](https://github.com/facebookresearch/llama)
2. [stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)
3. [alpaca-lora](https://github.com/tloen/alpaca-lora)
4. [Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html)
5.  [llama-7b-hf/tree/main](https://huggingface.co/decapoda-research/llama-7b-hf/tree/main)
6. [从0到1复现斯坦福羊驼（Stanford Alpaca 7B）](https://zhuanlan.zhihu.com/p/618321077)
7. [LLM系列 | 00：斯坦福 Alpaca 模型介绍及其复现](https://mp.weixin.qq.com/s/JgnyifW5ZKeK_sW8ig-wXw)
8. [ChatGPT平替模型：LLaMA（附下载地址，平民玩家和伸手党的福音！）](https://zhuanlan.zhihu.com/p/614118339)
9. [无需高性能GPU，在MacBook（或linux）上运行对标GPT3的LLaMA模型教程](https://www.bilibili.com/read/cv22383652?from=articleDetail)
