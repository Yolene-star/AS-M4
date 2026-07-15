# M4 项目架构与复现说明

这份文档面向“刚开始做计算机科研、希望彻底理解并修改 M4 模型”的读者。目标不是只解释 README，而是把代码里的真实实现、模型数据流、训练流程、推理逻辑、复现实验和可优化方向连成一个完整地图。

## 1. 一句话理解这个项目

M4，全称 Multimodal Multiplexing Modeling，是一个基于 LongVA / LLaVA-NeXT / Qwen2 的多模态交互模型框架。

它的核心目标不是重新训练一个大语言模型，而是在已有视觉语言模型上，用较小规模的指令数据，让模型具备更适合实时交互的能力：

1. 能看图像或视频。
2. 可接收文本问题。
3. 可扩展到语音问题。
4. 在模型正在回答时，能够判断新来的输入是有效追问、打断语句还是无须回应的噪声。
5. 通过并行解码和特殊 attention mask，让旧回答和新问题尽量不互相污染。

你可以把它理解成：

```text
普通 VLM：图像/视频 + 用户问题 -> 生成回答

M4：图像/视频流 + 旧问题 + 正在生成的回答 + 新来的问题/语音
    -> 判断新输入是否值得响应
    -> 如果值得响应，切换/并行处理新问题
    -> 如果是噪声，继续原回答
```

## 2. 项目结构地图

项目根目录中最重要的部分如下：

```text
M4-main/
├── README.md
├── requirements.txt
├── m4-it-qwen.json
├── voiceassistant.json
├── assets/
│   └── framework.png
├── preprocess/
│   └── tts/
├── VoiceAssistant-400K/
├── m4-it/
└── intersuit/
    ├── pyproject.toml
    ├── scripts/
    ├── local_demo/
    ├── checkpoints/
    ├── intersuit/
    │   ├── constants.py
    │   ├── conversation.py
    │   ├── mm_utils.py
    │   ├── utils.py
    │   ├── model/
    │   ├── train/
    │   └── eval/
    └── trl/
```

关键文件索引：

| 目标 | 关键文件 |
| --- | --- |
| 项目说明 | `README.md` |
| Python 包配置 | `intersuit/pyproject.toml` |
| 依赖列表 | `requirements.txt` |
| 视觉/语音融合主逻辑 | `intersuit/intersuit/model/llava_arch.py` |
| Qwen2 多模态语言模型 | `intersuit/intersuit/model/language_model/llava_qwen.py` |
| CLIP 视觉编码器 | `intersuit/intersuit/model/multimodal_encoder/clip_encoder.py` |
| 视觉 projector | `intersuit/intersuit/model/multimodal_projector/builder.py` |
| 视觉 resampler | `intersuit/intersuit/model/multimodal_resampler/builder.py` |
| Whisper 语音编码器 | `intersuit/intersuit/model/speech_encoder/speech_encoder.py` |
| 语音 projector | `intersuit/intersuit/model/speech_projector/speech_projector.py` |
| 训练入口 | `intersuit/intersuit/train/train.py` |
| Trainer | `intersuit/intersuit/train/llava_trainer.py` |
| M4 视觉训练脚本 | `intersuit/scripts/finetune_m4.sh` |
| M4 语音训练脚本 | `intersuit/scripts/finetune_m4_audio.sh` |
| 单机 4090 快速训练脚本 | `intersuit/scripts/run_m4_train_once_4090.sh` |
| 文本打断 demo | `intersuit/local_demo/turntaking_cli.py` |
| 语音打断 demo | `intersuit/local_demo/turntaking_audio_cli.py` |
| 主动视频推理 demo | `intersuit/local_demo/proactive_cli.py` |
| 流式视频推理实现 | `intersuit/local_demo/inference.py` |

## 3. 模型总体结构

M4 的主体是一个“多模态输入包装过的 Qwen2 Causal LM”。

实际模型结构可以概括为：

```text
                         ┌──────────────────────────────┐
                         │          Qwen2 LLM            │
                         │  LlavaQwenForCausalLM         │
                         │  hidden size = config.hidden  │
                         └──────────────▲───────────────┘
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             │                          │                          │
      text token embedding       image/video embeddings      speech embeddings
             │                          │                          │
        tokenizer                 CLIP VisionTower             Whisper Encoder
             │                          │                          │
      input_ids tokens        mm_projector + resampler       speech_projector
             │                          │                          │
             └──────────────拼接成 inputs_embeds──────────────┘
                                        │
                                  causal LM loss
```

也就是说，Qwen2 并不直接“读取图片文件或音频文件”。图片和音频会先被编码成一串连续向量，这些向量的维度被 projector 调整到 Qwen2 的 hidden size，然后像普通 token embedding 一样插入输入序列中。

## 4. 模态输入如何进入模型

### 4.1 文本输入

文本通过 Hugging Face tokenizer 编码成 `input_ids`。如果使用 Qwen2，对话模板是 ChatML 风格：

```text
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
用户内容
<|im_end|>
<|im_start|>assistant
助手回答
<|im_end|>
```

实现位置：

```text
intersuit/intersuit/train/train.py
    preprocess_qwen()
    preprocess_qwen_av()
```

训练时，用户输入部分的 label 会被置为 `IGNORE_INDEX`，只让模型学习 assistant 的回答。

### 4.2 图像输入

图像路径来自训练 JSON 的 `image` 字段。数据集读取时会：

1. 用 PIL 打开图片。
2. 根据 `image_aspect_ratio` 做预处理。
3. 如果是 `anyres`，会按网格切成适合 CLIP 的块。
4. 把文本里的 `<image>` 占位符保留下来。

实现位置：

```text
intersuit/intersuit/train/train.py
    LazySupervisedDataset.process_image()
    LazySupervisedDataset._get_item()
```

进入模型后：

```text
image tensor
  -> CLIPVisionTower
  -> mm_projector
  -> vision_resampler
  -> 替换 input_ids 中的 IMAGE_TOKEN_INDEX
```

实现位置：

```text
intersuit/intersuit/model/llava_arch.py
    encode_images()
    encode_multimodals()
    prepare_inputs_labels_for_multimodal()
```

### 4.3 视频输入

视频在这个项目中本质上是多帧图像序列。

训练或 demo 中会从视频抽帧：

```text
video file
  -> frames
  -> CLIP 逐帧编码
  -> frame patch embeddings
  -> flatten 成一长串视觉 token
```

在 demo 中默认最多 32 帧，每帧经过空间池化后大约使用 144 个视觉 token。并行解码里有硬编码：

```python
prefix_length = 13 + 32 * 144 + 1
```

这表示：

```text
系统提示 token 约 13 个
+ 32 帧 * 每帧 144 个视觉 token
+ 1 个额外边界 token
```

这个硬编码是后续优化时要特别注意的点。如果你改变帧数、池化比例或视觉 token 数，需要同步修改推理逻辑。

### 4.4 语音输入

语音输入来自训练 JSON 的 `speech` 字段，对应 wav 文件。数据读取时使用：

```text
intersuit/intersuit/utils.py
    process_audio_with_whisper()
```

模型侧流程：

```text
wav
  -> whisper log-mel / audio features
  -> Whisper encoder
  -> speech_projector
  -> 替换 input_ids 中的 SPEECH_TOKEN_INDEX
```

关键实现：

```text
intersuit/intersuit/model/llava_arch.py
    encode_speech()
    prepare_inputs_labels_for_multimodal_av()

intersuit/intersuit/model/speech_encoder/speech_encoder.py
    WhisperWrappedEncoder

intersuit/intersuit/model/speech_projector/speech_projector.py
    EncoderProjectorConcat
```

语音 projector 的实现很重要：

```text
连续 k 帧 Whisper 输出拼接
  -> Linear(encoder_dim * k, 2048)
  -> ReLU
  -> Linear(2048, LLM hidden size)
```

这里的 `k` 来自：

```text
speech_encoder_ds_rate，默认 5
```

它控制语音 token 下采样率。`k` 越大，语音 token 越少，速度更快，但细粒度语音信息可能损失更多。

## 5. 特殊 token 与 label mask

项目中有两个非常关键的伪 token：

```text
<image>  -> IMAGE_TOKEN_INDEX
<speech> -> SPEECH_TOKEN_INDEX
```

它们不是普通词。训练预处理时，tokenizer 临时把 `<image>`、`<speech>` 加进词表，随后再把它们替换成特殊负数 index：

```text
preprocess_qwen_av()
    <image>  -> IMAGE_TOKEN_INDEX
    <speech> -> SPEECH_TOKEN_INDEX
```

模型 forward 前会做替换：

```text
input_ids:
    [文本 token, IMAGE_TOKEN_INDEX, 文本 token, SPEECH_TOKEN_INDEX, 文本 token]

变成 inputs_embeds:
    [文本 embedding, 图像 embeddings, 文本 embedding, 语音 embeddings, 文本 embedding]
```

同时，图像 embedding 和语音 embedding 对应的 labels 会被填成：

```text
IGNORE_INDEX
```

含义是：模型不需要预测这些视觉/语音向量，只需要利用它们作为上下文去预测 assistant 文本。

这是理解多模态大模型训练的核心之一：

```text
多模态 encoder 学到的是“把外部模态翻译到 LLM embedding 空间”；
语言模型 loss 仍然是 next-token prediction。
```

## 6. 训练数据 M4-IT

README 中说明 M4-IT 有 9963 条指令，分成四类：

| 类别 | 数量 | 作用 |
| --- | ---: | --- |
| Original | 2624 | 保留基础视觉问答能力，防止灾难性遗忘 |
| Interleave | 2376 | 让图像和文本可以交错出现，不强制 `<image>` 一定在开头 |
| Noise | 2563 | 教模型识别“无须回应”的噪声输入 |
| Stop | 2500 | 教模型处理打断、停止类指令 |

本仓库根目录有一个 `m4-it-qwen.json`，样本格式类似：

```json
{
  "id": "1943968004",
  "image": "1943968004.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "Who is the author of this book?\nAnswer the question using a single word or phrase.<image>\n"
    },
    {
      "from": "gpt",
      "value": "Elliott Garber"
    }
  ]
}
```

音频版本会多一个 `speech` 字段：

```json
{
  "id": "000000240632",
  "image": "000000240632.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\n"},
    {"from": "human", "value": "<speech>\n"},
    {"from": "gpt", "value": "[0.280,0.194,0.628,0.824]"}
  ],
  "speech": ["000000240632_0.wav", "000000240632_1.wav"]
}
```

注意：音频文件和 `<speech>` 的顺序必须对应。代码里会按出现顺序依次替换。

## 7. 训练流程

### 7.1 总体训练流程

```text
JSON 样本
  -> LazySupervisedDataset 读取图片/视频/音频
  -> preprocess_qwen_av 生成 input_ids 和 labels
  -> DataCollator padding 成 batch
  -> 模型 forward 前替换 <image>/<speech>
  -> Qwen2 Causal LM 计算 loss
  -> Deepspeed 分布式训练
  -> 保存 checkpoint
```

对应代码：

```text
intersuit/intersuit/train/train.py
    LazySupervisedDataset
    DataCollatorForSupervisedDataset
    make_supervised_data_module()
    train()
```

### 7.2 视觉版 M4 训练

脚本：

```bash
cd intersuit
bash scripts/finetune_m4.sh
```

关键配置：

```text
model_name_or_path: checkpoints/LongVA-7B-Qwen2
data_path: inputs/texts/m4-it-qwen.json
image_folder: inputs/images/llava-next
vision_tower: checkpoints/clip-vit-large-patch14-336
mm_projector_type: mlp2x_gelu
mm_patch_merge_type: unires
model_max_length: 32000
learning_rate: 1e-5
num_train_epochs: 1
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
deepspeed: scripts/zero3.json
```

可训练部分：

```text
mm_tunable_parts = "mm_vision_tower,mm_mlp_adapter,mm_language_model"
```

含义：

1. 训练 CLIP vision tower。
2. 训练视觉 projector。
3. 训练语言模型。
4. 不涉及语音模块。

### 7.3 语音版 M4 训练

脚本：

```bash
cd intersuit
bash scripts/finetune_m4_audio.sh
```

关键配置：

```text
model_name_or_path: checkpoints/LongVA-7B-Qwen2-VoiceAssistant/checkpoint-3000
data_path: inputs/texts/m4-it-qwen-audio.json
speech_folder: inputs/speech/interinst
speech_encoder: checkpoints/whisper/large-v3.pt
model_max_length: 16384
gradient_accumulation_steps: 8
deepspeed: scripts/zero1.json
```

可训练部分：

```text
mm_tunable_parts = "speech_projector,mm_mlp_adapter,mm_language_model"
```

含义：

1. Whisper encoder 默认冻结。
2. 训练 speech_projector。
3. 训练视觉 projector。
4. 训练语言模型。

### 7.4 为什么先训 VoiceAssistant

README 建议在音频版 M4 前先用 VoiceAssistant-400K 做音频指令调优：

```bash
bash scripts/finetune_voiceassistant.sh
```

原因是 M4-IT 音频数据主要用于“交互行为建模”，规模不够大。如果模型还没有基本语音理解能力，直接拿 M4-IT 音频版训练，容易把问题混在一起：

```text
模型答错，可能是：
1. 语音听不懂；
2. 视觉-语音没对齐；
3. 交互策略没学会；
4. 输出格式没学会。
```

先做 VoiceAssistant 调优，可以让模型至少具备基础语音问答能力，再学习 M4 的打断/噪声/多路交互行为。

## 8. 推理流程

### 8.1 普通生成

普通多模态生成调用：

```text
LlavaQwenForCausalLM.generate()
```

流程：

```text
文本 input_ids + images/speeches
  -> prepare_inputs_labels_for_multimodal 或 prepare_inputs_labels_for_multimodal_av
  -> inputs_embeds
  -> Hugging Face generate
  -> output_ids
  -> tokenizer.decode
```

### 8.2 流式视频主动推理

主动推理 demo：

```bash
cd intersuit
python -m local_demo.proactive_cli \
  --model_path M4-LongVA-Qwen-7B \
  --frame_fps 1 \
  --video_file local_demo/assets/water.mp4
```

核心代码：

```text
intersuit/local_demo/inference.py
    LiveInfer
```

它做的事情是：

```text
每隔 1 / frame_fps 秒读取一帧
  -> 编码成 frame embedding
  -> 追加到 KV cache
  -> 根据 attention 找到模型关注的帧
  -> 如果当前帧和用户需求相关，触发回答
```

这里有一个 attention-based grounding 逻辑：

```text
取最后一层 attention
  -> 对 head 求平均
  -> 看 query token 对历史视觉 token 的注意力
  -> 聚合到帧级别
  -> mean + 1.5 * std 作为阈值
  -> 超过阈值的帧认为是 salient frame
```

这不是一个额外分类器，而是利用语言模型自身的 attention 来估计“什么时候该回应”。

### 8.3 M4 并行打断推理

文本打断 demo：

```bash
cd intersuit
python -m local_demo.turntaking_cli \
  --video_path local_demo/assets/water.mp4 \
  --question "Can you describe the video?" \
  --new_query "How many people in the video?" \
  --new_query_pos 20
```

语音打断 demo：

```bash
cd intersuit
python -m local_demo.turntaking_audio_cli \
  --video_path local_demo/assets/water.mp4 \
  --question "Can you describe the video?" \
  --new_query "How many people in the video?" \
  --new_query_pos 20
```

核心函数：

```text
intersuit/intersuit/model/language_model/llava_qwen.py
    generate_parallel()
```

逻辑简化如下：

```text
1. 模型开始回答原问题。
2. 生成到第 new_query_pos 个 token 时，新问题插入。
3. 构造新的 channel 序列：
      prefix / video 属于公共上下文
      原回答属于 channel 1
      新问题属于 channel 2
4. 构造 transition attention mask：
      所有 channel 都能看公共 prefix
      每个 channel 只能看自己 channel 内的历史 token
      保持 causal mask，不看未来 token
5. 用新问题最后一个位置的 logits 判断它是不是 noise。
6. 如果是噪声，裁剪 KV cache，继续旧回答。
7. 如果不是噪声，切换到新问题继续生成。
```

这就是 M4 “multiplexing” 的关键。

## 9. 并行 attention mask 的直观解释

普通 causal attention：

```text
token i 可以看所有 j <= i 的 token
```

M4 并行 attention：

```text
公共前缀：所有分支都可以看
同一分支：可以看历史 token
不同分支：不能互相看
未来 token：不能看
```

代码中的判断：

```python
combined_mask = (prefix_mask | block_mask) & causal_mask
```

含义：

```text
prefix_mask：允许看公共上下文
block_mask：允许看同一 channel 的上下文
causal_mask：不允许看未来
```

为什么这么做？

假设模型正在回答：

```text
用户 A：描述视频
助手 A：视频中有一条河...
```

突然来了一个新输入：

```text
用户 B：How many people are in the video?
```

如果直接拼到普通 causal 序列后面，新问题会看到旧回答的全部 token，可能被旧回答干扰。M4 希望新问题共享视频上下文，但不被旧回答污染，所以把旧回答和新问题放到不同 channel。

## 10. 噪声判断机制

在 `generate_parallel()` 中，新问题插入后，代码会检查：

```python
noise_prob = next_prob[:, 151644]
```

注释写的是：

```text
151644 -> <|im_start|>
```

也就是说，模型被训练成：当输入是噪声或无须回应内容时，更倾向输出一个特殊标志 token。代码用该 token 的概率与动态阈值比较：

```python
next_entropy = -torch.sum(next_prob * torch.log(next_prob + 1e-5), dim=-1)
noise_threshold = min(0.09, exp(-entropy) * 0.3)
```

直观理解：

```text
如果模型很确定“这是噪声”，noise token 概率会高。
如果整体分布很不确定，则阈值也会调整。
```

注意：这里有硬编码 token id `151644`。如果更换 tokenizer 或模型，这个 id 可能不再代表 `<|im_start|>`。优化时建议改成：

```python
noise_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
```

## 11. 复现环境要求

README 中测试环境：

```text
Python: 3.10
CUDA: 11.8
GPU: A800-80G
PyTorch: 2.5.0
torchvision: 0.20.0
torchaudio: 2.5.0
transformers: 4.44.0
deepspeed: 0.14.5
flash-attn: 2.6.3
```

推荐安装：

```bash
conda create -n open_gpt4o python=3.10 -y
conda activate open_gpt4o

pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
  --index-url https://download.pytorch.org/whl/cu118

cd intersuit
pip install -e ".[train]"

cd ..
pip install packaging ninja
pip install flash-attn==2.6.3 --no-build-isolation --no-cache-dir
pip install -r requirements.txt
```

如果只是调试代码逻辑，不一定需要 A800。但如果要完整训练 7B 模型，显存压力很大。

## 12. 数据和 checkpoint 目录要求

README 期望数据组织如下：

```text
intersuit/inputs/
├── images/
│   └── llava-next/
├── speech/
│   ├── voiceassistant/
│   └── interinst/
└── texts/
    ├── voiceassistant.json
    ├── m4-it-qwen.json
    └── m4-it-qwen-audio.json
```

checkpoint 期望如下：

```text
intersuit/checkpoints/
├── LongVA-7B-Qwen2/
├── M4-LongVA-7B-Qwen2/
├── clip-vit-large-patch14-336/
└── whisper/
    └── large-v3.pt
```

本仓库已经有一些 checkpoint 文件和数据样例，但你复现实验时仍要确认：

1. 权重文件是否完整。
2. JSON 中的图片名是否能在 `image_folder` 下找到。
3. JSON 中的音频名是否能在 `speech_folder` 下找到。
4. tokenizer 与模型 config 是否匹配。

## 13. 新手推荐复现路线

不要一开始就完整训练。建议按下面顺序走：

### 阶段 1：只跑 demo 推理

目的：确认模型、tokenizer、视觉编码器、视频读取都能正常工作。

```bash
cd intersuit
python -m local_demo.turntaking_cli \
  --video_path local_demo/assets/water.mp4 \
  --question "Can you describe the video?" \
  --new_query "How many people in the video?" \
  --new_query_pos 20
```

如果失败，优先查：

```text
1. CUDA 是否可用。
2. checkpoint 路径是否正确。
3. transformers / torch 版本是否兼容。
4. attn_implementation 是否要从 flash_attention_2 改为 sdpa/eager。
```

### 阶段 2：跑 1 step 训练

本仓库有一个适合快速检查的脚本：

```bash
cd intersuit
MAX_STEPS=1 bash scripts/run_m4_train_once_4090.sh
```

目的不是训练出好模型，而是验证：

```text
数据读取 -> 图像预处理 -> tokenizer -> forward -> backward -> save checkpoint
```

### 阶段 3：小数据过拟合

从 `m4-it-qwen.json` 抽 20-100 条样本，训练几十步，看 loss 是否下降。

如果 loss 不下降，优先检查：

```text
1. labels 是否全是 IGNORE_INDEX。
2. <image>/<speech> 是否被正确替换。
3. projector 是否参与训练。
4. learning rate 是否太小。
5. 数据路径是否错位。
```

### 阶段 4：完整 M4-IT 微调

确认小数据正常后，再跑：

```bash
cd intersuit
bash scripts/finetune_m4.sh
```

或音频版：

```bash
cd intersuit
bash scripts/finetune_m4_audio.sh
```

## 14. 实验结果如何解读

这个仓库 README 没有给详细分数表，只说明推荐用 OmniMMI 评测 streaming video context 中的交互能力。因此你解读实验时，不应只看普通 VQA 准确率，而要分维度看。

建议至少评估以下指标：

| 能力 | 观测方式 |
| --- | --- |
| 基础视觉理解 | 图像/视频问答是否正确 |
| 原始能力保持 | Original replay 样本上是否退化 |
| 新问题响应 | 有效 new query 是否被切换并回答 |
| 噪声过滤 | “Okay, I see.” 这类噪声是否被忽略 |
| 打断执行 | “Can I stop you?” 是否停止/结束当前输出 |
| 延迟 | new query 插入后多久开始新回答 |
| 流式稳定性 | 长视频持续输入时是否 cache 爆掉或重复输出 |
| 音频理解 | 文本问题改成语音后性能掉多少 |

对于 `generate_parallel()`，最重要的实验不是 BLEU/ROUGE，而是三类新输入的行为：

```text
有效问题：应该切换回答。
打断指令：应该停止或改变当前输出。
噪声语句：应该继续原回答。
```

README 中给出的 demo 正好对应这三类：

```bash
# 有效追问
--new_query "How many people in the video?"

# 打断语句
--new_query "Sorry to interrupt?"

# 噪声语句
--new_query "Okay, I see."
```

## 15. 重要超参数解释

| 参数 | 位置 | 含义 | 修改影响 |
| --- | --- | --- | --- |
| `model_max_length` | 训练脚本 | 最大序列长度 | 越大越能容纳视频/长对话，但显存暴涨 |
| `mm_projector_type` | 训练脚本 / builder | 视觉特征投影结构 | 影响视觉特征对齐能力 |
| `mm_patch_merge_type` | 训练脚本 / llava_arch | 图像 patch 合并方式 | 影响视觉 token 数和细节 |
| `mm_vision_select_layer` | CLIP encoder | 选 CLIP 第几层特征 | 不同层语义/细节不同 |
| `mm_spatial_pool_stride` | demo config | 视频帧 token 池化比例 | 越大越快但细节少 |
| `speech_encoder_ds_rate` | speech projector | 语音下采样率 | 越大语音 token 越少 |
| `new_query_pos` | demo | 第几个生成 token 插入新问题 | 用于模拟实时打断 |
| `noise_threshold` | generate_parallel | 噪声判断阈值 | 影响误拒/误接新问题 |
| `prefix_length` | generate_parallel | 公共上下文长度 | 若视频 token 数变动必须同步 |

## 16. 最值得优化的方向

### 16.1 去掉硬编码 prefix_length

当前代码：

```python
prefix_length = 13 + 32 * 144 + 1
```

问题：

1. 帧数不一定是 32。
2. 每帧 token 数不一定是 144。
3. system prompt token 数可能变。
4. 改 tokenizer 后 token 数可能变。

建议：

```text
在 prepare_inputs_labels_for_multimodal 后记录：
1. system prompt 长度
2. image/video embedding 插入区间
3. user query 起止位置
然后动态计算 prefix_length。
```

改动入口：

```text
intersuit/intersuit/model/language_model/llava_qwen.py
    generate_parallel()
```

### 16.2 去掉硬编码 noise token id

当前代码：

```python
noise_prob = next_prob[:, 151644]
```

建议改成：

```python
noise_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
noise_prob = next_prob[:, noise_token_id]
```

进一步优化：不要只靠一个 token，可以训练一个显式分类头：

```text
hidden state
  -> noise / valid / stop 三分类
```

这样可解释性更强，也便于调阈值。

### 16.3 为打断类型增加显式标签

M4-IT 当前通过目标输出让模型隐式学习：

```text
有效问题 -> 正常回答
噪声 -> 特殊 token
停止 -> <|im_end|>
```

可以改成多任务学习：

```text
主任务：生成回答
辅助任务：classify(new_input) = valid / noise / stop
```

好处：

1. 更容易调试。
2. 不依赖特殊 token 概率。
3. 可以单独评测分类准确率。

### 16.4 改进 speech_projector

当前语音 projector 是简单 concat + MLP：

```text
k 帧拼接 -> Linear -> ReLU -> Linear
```

可尝试：

1. 加 LayerNorm。
2. 改成 Q-Former / Perceiver Resampler。
3. 加 cross-attention，把语音特征压缩成固定数量 token。
4. 使用 CTC/ASR 辅助损失稳定语音理解。

改动入口：

```text
intersuit/intersuit/model/speech_projector/speech_projector.py
intersuit/intersuit/model/speech_projector/builder.py
```

### 16.5 视频 token 压缩

视频长了以后 token 数很容易爆炸：

```text
32 frames * 144 tokens/frame = 4608 visual tokens
```

可优化方向：

1. 只保留 salient frames。
2. 对相邻帧做 temporal pooling。
3. 引入 memory token。
4. 用 lightweight temporal transformer 压缩。
5. 动态调整 frame_fps。

改动入口：

```text
intersuit/local_demo/inference.py
intersuit/intersuit/model/llava_arch.py
intersuit/intersuit/model/multimodal_resampler/
```

### 16.6 更系统的评测

建议建立一个小型本地 benchmark：

```text
每条样本包含：
video/image
original_question
new_query
new_query_type: valid / noise / stop
expected_behavior
reference_answer
```

指标：

```text
valid switch accuracy
noise rejection accuracy
stop success rate
answer relevance
latency
```

这样你每次改模型都能判断是否真的变好。

## 17. 常见坑

### 17.1 label 全被 mask

如果训练 loss 异常小或不动，检查 labels。正常情况下：

```text
user/system/image/speech 部分是 IGNORE_INDEX
assistant 回答部分不是 IGNORE_INDEX
```

### 17.2 图像数量和 `<image>` 数量不一致

如果一个样本里有多个 `<image>`，代码会按出现顺序消耗 image_features。数量不一致会导致错位或报错。

### 17.3 speech 数量和 `<speech>` 数量不一致

同上，音频列表顺序必须与 `<speech>` 出现顺序一致。

### 17.4 tokenizer 特殊 token 变化

`IMAGE_TOKEN_INDEX` 和 `SPEECH_TOKEN_INDEX` 是内部特殊 index，不是 tokenizer 原生 id。不要直接拿普通 tokenizer id 替代它们。

### 17.5 更换 base model

如果从 Qwen2 换成 Llama/Mistral：

1. 对话模板要换。
2. special token id 会变。
3. `preprocess_*` 函数要匹配。
4. `generate_parallel()` 中硬编码 token 需要检查。

## 18. 建议阅读代码顺序

如果你是新手，推荐按这个顺序读：

1. `README.md`
2. `intersuit/scripts/finetune_m4.sh`
3. `intersuit/intersuit/train/train.py` 中的 dataclass 参数定义
4. `LazySupervisedDataset._get_item()`
5. `preprocess_qwen_av()`
6. `DataCollatorForSupervisedDataset.__call__()`
7. `intersuit/intersuit/model/llava_arch.py`
8. `intersuit/intersuit/model/language_model/llava_qwen.py`
9. `generate_parallel()`
10. `intersuit/local_demo/inference.py`

读代码时重点问自己：

```text
这个张量现在是什么 shape？
它代表文本 token、图像 token 还是语音 token？
它参与 loss 吗？
它能 attend 到哪些历史 token？
它是否进入 KV cache？
```

## 19. 最小闭环总结

M4 的核心闭环是：

```text
数据：
    M4-IT 构造 original / interleave / noise / stop 四类交互样本

编码：
    CLIP 编图像/视频
    Whisper 编语音
    projector 映射到 Qwen2 hidden space

融合：
    用图像/语音 embeddings 替换 <image>/<speech> 占位符

训练：
    只对 assistant 文本做 causal LM loss

推理：
    普通问题正常 generate
    新问题插入时构造 parallel attention mask
    通过特殊 token 概率判断 noise
    valid 则切换回答，noise 则继续旧回答
```

如果你要优化它，优先从这几个点下手：

```text
1. 动态 prefix_length
2. 动态 noise token id
3. 显式 valid/noise/stop 分类头
4. 更强 speech_projector
5. 视频 token 压缩
6. 本地可重复 benchmark
```

这几个方向的风险和收益都比较清晰，非常适合作为科研小白的第一轮改进实验。
