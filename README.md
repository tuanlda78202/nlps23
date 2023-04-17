# Vietnamese Poem Generator
![Vietnam Reunification Day 30-4-1975](https://www.google.com/url?sa=i&url=https%3A%2F%2Ftime.com%2F3840657%2Fsaigon-fall-lessons%2F&psig=AOvVaw3RO_x59-UB3Nwcd8ySiSNy&ust=1681838577368000&source=images&cd=vfe&ved=0CBAQjRxqFwoTCKC_94m3sf4CFQAAAAAdAAAAABAo)

This is the source code of the project "Vietnamese Poem Generator" of the course "Natural Language Processing" Summer 2023.

---
- [Vietnamese Poem Generator](#vietnamese-poem-generator)
  - [Abstract](#abstract)
  - [Folder Structure](#folder-structure)
  - [Model Zoo](#model-zoo)
  - [Usage](#usage)
    - [Config file format](#config-file-format)
    - [Using config files](#using-config-files)
    - [Resuming from checkpoints](#resuming-from-checkpoints)
    - [Evaluating](#evaluating)
    - [Inference](#inference)
  - [Contributors](#contributors)
## Abstract 
Vietnamese poetry has a rich history that dates back to the 10th century, with the emergence of the "Lục Bát" (six-eight) form, which is characterized by alternating lines of six and eight syllables. Since then, Vietnamese poetry has undergone various transformations and has been influenced by different cultural and historical periods, such as Chinese Confucianism, French colonialism, and modernization. Vietnamese poetry often includes themes of love, nature, patriotism, and social issues, and follows strict rules and structures, such as tonal patterns, rhyming schemes, and word choice.

This project proposes a Vietnamese poem generator that utilizes a combination of state-of-the-art natural language processing techniques, including diffusion, transformer, GPT, and LLMs. The proposed model generates high-quality Vietnamese poems that adhere to the traditional rules and structures of Vietnamese poetry while also incorporating modern themes and language. The generator is based on a large corpus of Vietnamese poetry and uses the diffusion technique to enhance the coherence and fluency of the generated poems. The transformer-based architecture is used for encoding and decoding, while the GPT and LLMs techniques are employed for language modeling and improving the diversity of the generated poems. 

The performance of the proposed model is evaluated through a set of quantitative and qualitative metrics, including perplexity, rhyme, and coherence. The experimental results demonstrate the effectiveness of the proposed model in generating high-quality Vietnamese poems that are both linguistically and aesthetically pleasing. The proposed model has potential applications in various fields, including literature, education, and art.

## Folder Structure

```
nlps23/
├── configs/ - training config
|   ├── README.md - config name style
│   ├── */README.md - abstract and experiment results model
|   ├── api-key/ - wandb api key for monitoring
|
├── tools/ - script to downloading data, training, testing, inference and web interface
|
├── trainer/ - trainer classes 
|
├── model/ 
|   ├── architecture/ - model architectures
|   ├── README.md - losses and metrics definition
|
├── base/ - abstract base classes
│   
├── data/ - storing input data
|
├── data_loader/ - custom dataset and dataloader
│
├── saved/ - trained models config, log-dir and logging output
│
├── logger/ - module for wandb visualization and logging
|
├── utils/ - utility functions
```
## Model Zoo 
<summary></summary>

<table style="margin-left:auto;margin-right:auto;font-size:1.4vw;padding:10px 10px;text-align:center;vertical-align:center;">
  <tr>
    <td colspan="5" style="font-weight:bold;">Salient Object Detection</td>
  </tr>
  <tr>
    <td><a href="https://github.com/tuanlda78202/CVP/blob/main/configs/mobilenetv3/README.md">MobileNetV3</a> (ICCV'2019)</td>
    <td><a href="https://github.com/tuanlda78202/CVP/blob/main/configs/u2net/README.md">U2Net</a> (PR'2020)</td>
    <td><a href="https://github.com/tuanlda78202/CVP/blob/main/configs/bisenet/README.md">BiSeNet</a> (CVPR'2021)</td>
    <td><a href="https://github.com/tuanlda78202/CVP/blob/main/configs/dis/README.md">DIS</a> (ECCV'2022)</td>
    <td><a href="https://github.com/tuanlda78202/CVP/blob/main/configs/inspyrenet/README.md">InSPyReNet</a> (ACCV'2022)</td>
  </tr>

</table>

## Usage

Install the required packages:

```
pip install -r requirements.txt
```
<!-- pipreqs for get requirements.txt -->

Running private repository on Kaggle:
1. [Generate your token](https://github.com/settings/tokens)
2. Get repo address from `github.com/.../...git`: 
```bash
git clone https://your_personal_token@your_repo_address.git
cd CVP
```
### Config file format

<details>
<summary>Config files are in YAML format</summary>

```yaml
name: U2NetFull_scratch_1gpu-bs4_KNC_size320x320

n_gpu: 1

arch:
  type: u2net_full
  args: {}

data_loader:
  type: KNC_DataLoader
  args:
    batch_size: 4
    shuffle: true
    num_workers: 1
    validation_split: 0.1
    output_size: 320
    crop_size: 288

optimizer:
  type: Adam
  args:
    lr: 0.001
    weight_decay: 0
    eps: 1.e-8
    betas:
      - 0.9
      - 0.999

loss: multi_bce_fusion

metrics:
  - mae
  - sm

lr_scheduler:
  type: StepLR
  args:
    step_size: 50
    gamma: 0.1

trainer:
  type: Trainer

  epochs: 1000
  save_dir: saved/
  save_period: 10
  verbosity: 1

  visual_tool: wandb
  project: cvps23
  name: U2NetLite_scratch_1gpu-bs4_KNC_size320x320

  # Edit *username for tracking WandB multi-accounts
  api_key_file: ./configs/api-key/tuanlda78202
  entity: tuanlda78202
  
test:
  save_dir: saved/generated
  n_sample: 1000
  batch_size: 32
```

</details>

### Using config files
Modify the configurations in `.yaml` config files, then run:

```bash
python tools/train.py [CONFIG] [RESUME] [DEVICE] [BATCH_SIZE] [EPOCHS]
```

### Resuming from checkpoints
You can resume from a previously saved checkpoint by:

```bash
sh tools/train.py --resume path/to/the/ckpt
```

### Evaluating
```bash
python tools/eval.py
```

### Inference 
- Running demo on notebook [`inference.ipynb`](https://github.com/tuanlda78202/cvps23/blob/main/tools/inference.ipynb) in [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tuanlda78202/CVP/)

- Web Interface: Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/doevent/dis-background-removal) <br> 


## Contributors 
<!-- https://contrib.rocks/preview?repo=tuanlda78202%2FCVP -->

<a href="https://github.com/tuanlda78202/CVP/graphs/contributors">
<img src="https://contrib.rocks/image?repo=tuanlda78202/CVP" /></a>
<a href="https://github.com/tuanlda78202/CVP/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=tuanlda78202/CVP" />
</a>
