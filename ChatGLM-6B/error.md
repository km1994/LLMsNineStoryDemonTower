# ChatGLM-6B 小编填坑记

## 一、动机

清华 开源了 ChatGLM-6B 大模型，效果是真的香，但是 由于 版本的迭代，导致 小编在 复现 ChatGLM-6B  关联的项目（eg:ChatGLM-6B、P-Tuning v2 、mymusise/ChatGLM-Tuning、yongzhuo/chatglm-maths等）一路碰壁，真的是 “复现三分钟，调试一天半”。

至此，小编为了 不然 后来者 再次 踩坑，特出该 《填坑笔记》，已帮助大佬们渡过难关。

后期会不断更新。

## 二、小编填坑记

### 2.1 显卡不够怎么办

- 问题描述：如果你的 GPU 显存有限怎么办？
- 解决方法：可以尝试以量化方式加载模型

ChatGLM 提供 三种量化方式：

1. FP16 精度加载

```python
    # 按需修改，目前只支持 4/8 bit 量化
    model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
```

2. 4/8 bit 量化

```python
    # 按需修改，目前只支持 4/8 bit 量化
    model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().quantize(4).cuda()
```

> 注：8-bit 量化下 GPU 显存占用约为 10GB，4-bit 量化下仅需 6GB 占用


### 2.2 RuntimeError: Internal: [MASK] is already defined.

> 参考：[RuntimeError: Internal: [MASK] is already defined.](https://github.com/hscspring/hcgf/issues/14)

- 问题描述：使用 ChatGLM-6B 报错

最新的不行，用上一个版本的chatglm-6b 可以

```s
    $ RuntimeError: Internal: [MASK] is already defined.
```

- 解决方法：

1. clone THUDM/chatglm-6b

```s
    $ git clone https://huggingface.co/THUDM/chatglm-6b
    >>>
    Cloning into 'chatglm-6b'...
    remote: Enumerating objects: 372, done.
    remote: Counting objects: 100% (369/369), done.
    remote: Compressing objects: 100% (153/153), done.
    remote: Total 372 (delta 229), reused 348 (delta 216), pack-reused 3
    Receiving objects: 100% (372/372), 103.83 KiB | 2.00 MiB/s, done.
    Resolving deltas: 100% (229/229), done.
    Filtering content: 100% (9/9), 12.49 GiB | 734.00 KiB/s, done.
```

2. 切换到到 commit_id 为 55 版本

```s
    $ git reset --hard 551a50efec3acc5a9b94de8ec46d33d0f81919f7
```

### 2.3 ValueError: 130004 is not in list

- 问题描述

在 跑 [mymusise/ChatGLM-Tuning](https://github.com/mymusise/ChatGLM-Tuning) 后，运行一下命令，

```s
    $ python finetune.py \ --dataset_path /output/train \ --lora_rank 8 \ --per_device_train_batch_size 6 \ --gradient_accumulation_steps 1 \ --max_steps 3000 \ --save_steps 1000 \ --save_total_limit 2 \ --learning_rate 1e-4 \ --fp16 \ --remove_unused_columns false \ --logging_steps 50 \ --output_dir /output/lora
```

之后出现以下问题

```s
    $ the error is "ValueError: 130004 is not in list"
```

- 问题原因：
  - ChatGLM-6B 的 huggingface repo 更新了，需要重新下载模型下来，然后再运行（官方的一些特殊 token 的 ID 又变了）
  - 这个问题在输入内容长度太长时会出现，原因是在encode时先做了特殊token的拼接再截断，导致截断后，拼接的特殊token 150001 150004丢掉了
- 解决方法：

1. 直接在tokenize_dataset_rows.py preprocess函数中将 prompt_ids处理下，把最后2个id改为150001 150004

```s
    def preprocess(tokenizer, config, example, max_seq_length):
        prompt = example["context"]
        target = example["target"]
        prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
        # 加了 下面的 。。。。。。。
        prompt_ids[-2] = 150001
        prompt_ids[-1] = 150004
        # 加了 上面的 然后重新生成数据 。。。。。。。
        target_ids = tokenizer.encode(
            target,
            max_length=max_seq_length,
            truncation=True,
            add_special_tokens=False)
        input_ids = prompt_ids + target_ids + [config.eos_token_id]
        
        return {"input_ids": input_ids, "seq_len": len(prompt_ids)}
```

2. 然后重新生成数据 重新运行

```s
    $ python tokenize_dataset_rows.py --jsonl_path data/alpaca_data.jsonl --save_path data/alpaca --max_seq_length 200 --skip_overlength false
```

### 2.4 no module named 'torch._six'

- 问题描述：安装 torch2.0 版本后出现问题
- 解决方法：将 torch 降级到 1.13.1 版本 即可

```s
    $ pip install torch==1.13.1
```

### 2.5 抛出异常 No module named 'transformers_modules.' 

- 问题描述：

在执行单元格：

```s
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
```

抛出异常 

```s
    No module named 'transformers_modules.'
```

- 解决方法：切换transformers成4.27.1

```s
    $ pip install transformers==4.27.1
```

### 2.6 AttributeError: 'ChatGLMForConditionalGeneration' object has no attribute 'enable_input_require_grads'

- 问题描述：

```s
    AttributeError: 'ChatGLMForConditionalGeneration' object has no attribute 'enable_input_require_grads'
```

- 解决方法：更新transformers 库 就行, 这个在4.27 应该就有了 peft那移植过来的

### 2.7 低秩自适应LORA, RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!

- 问题描述

```s
    低秩自适应LORA, RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

- 解决方法：尝试 transformers升级到最新, get_peft_model后再.cuda(), device_map={'':torch.cuda.current_device()}

### 2.8 ChatGLMTokenizer有时候会报奇奇怪怪的错误

- 问题描述

```s
    ChatGLMTokenizer有时候会报奇奇怪怪的错误
```

- 解决方法：生成时候设置max_new_tokens, 最大{"max_new_tokens": 2048}; decode有时候会出现不存在id;

## 三、总结

小编一口气 重新 填坑了 [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) 几大项目中的常规大坑，欢迎大佬们 绕坑！


## 参考

1. [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) 
2. [P-Tuning v2](https://github.com/THUDM/P-tuning-v2) 
3. [mymusise/ChatGLM-Tuning](https://github.com/mymusise/ChatGLM-Tuning)
4. [yongzhuo/chatglm-maths](https://github.com/yongzhuo/chatglm-maths)

