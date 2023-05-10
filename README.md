# LLMsNineStoryDemonTower LLMs九层妖塔
【LLMs九层妖塔】分享一下打怪(ChatGLM、Chinese-LLaMA-Alpaca、MiniGPT-4、FastChat、LLaMA、gpt4all等)实战与经验，

![LLMs九层妖塔 视频介绍](https://github.com/km1994/LLMsNineStoryDemonTower/blob/main/mp4/LLMs九层妖塔挑战赛.mp4)
> [LLMs九层妖塔 视频介绍 地址](https://github.com/km1994/LLMsNineStoryDemonTower/blob/main/mp4/LLMs九层妖塔挑战赛.mp4)

- [LLMsNineStoryDemonTower LLMs九层妖塔](#llmsninestorydemontower-llms九层妖塔)
  - [【LLMs 入门实战系列】](#llms-入门实战系列)
    - [第一层 ChatGLM-6B](#第一层-chatglm-6b)
      - [内容大纲](#内容大纲)
    - [第二层 Stanford Alpaca 7B](#第二层-stanford-alpaca-7b)
    - [第三层 Chinese-LLaMA-Alpaca](#第三层-chinese-llama-alpaca)
    - [第四层 小羊驼 Vicuna](#第四层-小羊驼-vicuna)
    - [第五层 MiniGPT-4](#第五层-minigpt-4)
    - [第六层 GPT4ALL](#第六层-gpt4all)
    - [第七层 AutoGPT](#第七层-autogpt)
    - [第八层 MOSS](#第八层-moss)
  - [优秀笔记](#优秀笔记)
    - [第一层](#第一层)
    - [优秀笔记](#优秀笔记-1)
  - [参考](#参考)

## 【LLMs 入门实战系列】

### 第一层 ChatGLM-6B

#### 内容大纲

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

## 优秀笔记

### 第一层

### 优秀笔记

1. [杨夕](https://mp.weixin.qq.com/s/4QNgF6nAUo8imSaIB_OWmg)
2. [奔腾](https://articles.zsxq.com/id_k2qzsps7zw21.html)
3. [逸尘](https://articles.zsxq.com/id_zzfqt88sw4rl.html)
4. [此方一泉](https://t.zsxq.com/0dEp8PDcW)
5. [vezel](http://t.csdn.cn/hWn9D)
6. [徐生](https://zhuanlan.zhihu.com/p/627358709)
7. [多点微笑](https://articles.zsxq.com/id_velwvtmfhrwz.html)
8. [小固](https://zhuanlan.zhihu.com/p/627333187)
9. [土狼](https://zhuanlan.zhihu.com/p/627358709)
10. [0](https://github.com/Wesley12138/LLM)
11. [Welch](https://t.zsxq.com/0dJhaaGRW)
12. [九猫](https://articles.zsxq.com/id_7g0g65fbsluo.html)


## 参考

1. [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)
2. [Stanford Alpaca 7B](https://github.com/tatsu-lab/stanford_alpaca)
3. [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
4. [Vicuna](https://github.com/lm-sys/FastChat)
5. [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)
6. [GPT4ALL](https://github.com/nomic-ai/gpt4all)
7. [Auto-GPT](hhttps://github.com/Significant-Gravitas/Auto-GPT)
8. [MOSS](https://github.com/OpenLMLab/MOSS/tree/main)
