# PhantomSeal: Proactive Deepfakes Defense with Identity/Context Protection and Forensic Tracing

---

**Abstract**: Deepfakes, especially face-swapping attacks, pose significant challenges to authenticity, security, and ethics across science, engineering, and society. While most existing detection/tracing approaches operate post hoc, proactive defenses that aim to intervene before deepfake generation remain limited in terms of real-world effectiveness. In this paper, we present PhantomSeal, the first proactive defense to simultaneously protect both the identity and the context of user images from being used in face-swapping attacks, while supporting forensic tracing. We present a novel cloaking technique that embeds a selected identity as a stealthy identifier. This mechanism steers the deepfake generation process toward producing content that resembles the chosen cloak identity, thereby preventing successful face-swapping while enabling effective feature-based forensic analysis. The effectiveness and robustness of PhantomSeal is demonstrated in extensive experiments across different face-swapping architectures and models. For example, it reduces the attack success rate of SimSwap, an advanced deepfake model, to 0.30%, and correctly identifies 97.97% of manipulated content.

![PhantomSeal overview](assets/phantomseal.png)

---

## Contents

- [PhantomSeal: Proactive Deepfakes Defense with Identity/Context Protection and Forensic Tracing](#phantomseal-proactive-deepfakes-defense-with-identitycontext-protection-and-forensic-tracing)
  - [Contents](#contents)
  - [Description](#description)
  - [Getting Started](#getting-started)
    - [Environment](#environment)
    - [Dataset, Pre-Trained Models, and Third-Party Projects](#dataset-pre-trained-models-and-third-party-projects)
  - [Usage](#usage)

## Description

This repository is the official implementation of PhantomSeal, a framework for protecting both image identity and contextual integrity, while enabling forensic traceability. The repository provides the code necessary to reproduce all experimental results reported in the paper. All random seeds are fixed to 0 to improve experimental stability. Nevertheless, due to inherent nondeterminism in GPU-based computation and iterative optimization, minor variations in outputs may still occur.

We implemented PhantomSeal using Python 3.10.19, PyTorch 2.8, and CUDA 12.8. The experiments reported in the paper were conducted on a workstation running Ubuntu 24.04 LTS with an AMD Ryzen 9 9950X 16-core CPU, an NVIDIA RTX 5090 GPU, and 64 GB memory. The reported results are mainly evaluated with FaceNet-512 and Face++; Face Recognition and AWS Rekognition are also supported as additional evaluation tools. Because Face++ and AWS Rekognition are commercial paid APIs, the open-source code evaluates experiments with the open-source FaceNet-512 and Face Recognition tools by default. To use paid APIs, configure the corresponding key/secret in **config/evaluate/evaluate_local.yaml**.

## Getting Started

### Environment

We strongly recommend using Conda to manage the environment. We provide **environment.yml** to create the default *phantomseal* environment. Please use the following commands to create and activate it.

```shell
conda env create -f environment.yml
conda activate phantomseal
```

A Docker-based environment is under preparation and will be added in a future update.

### Dataset, Pre-Trained Models, and Third-Party Projects

PhantomSeal is a defense framework. The face-swapping models used for evaluation are implemented by their corresponding third-party projects, and this repository integrates them only as evaluation backbones. The datasets, pre-trained models, and third-party project code remain owned by their original authors and are subject to their original licenses and terms.

To facilitate reproduction, we provide setup scripts that initialize the required third-party projects, apply local compatibility patches, and download the datasets and pre-trained model checkpoints from Google Drive. The datasets require about 14 GB of storage, and the checkpoints require about 14 GB. Depending on network conditions, the download process may take up to 20 minutes. Please use the following command to prepare the third-party projects, datasets, and checkpoints.

```shell
bash tools/setup.sh
```

If the automatic download fails, please visit the [Google Drive](https://drive.google.com/drive/folders/1caHioBnA1478FR15W3JzNuHu36zxopJv?usp=sharing), download the required files manually, unzip them, and place them in the repository root folder.

## Usage

All projects are controlled by their corresponding configuration files in *config/third_party*. All running results, including logs, metrics, and saved images, are written to the *logs* folder. Single-run experiments are saved under *logs/run*, while multi-run experiments with multiple parameter settings are saved under *logs/multirun*. In both cases, results are first grouped by date and then by run time. Evaluation runs are processed batch by batch. For each batch, PhantomSeal reports both the current batch result and the cumulative summary up to that batch, so users can monitor the running summary and stop early when enough samples have been evaluated to save time.

Most experiments evaluate 3,000 images by default, and the running time depends on the actual number of images being evaluated. The reported results and expected running times are based on our experimental environment. On different hardware, especially different GPU environments, users may need to adjust the corresponding *batch_size* values in the configuration files to fit available memory and throughput.

We provide project scripts in the *scripts* folder for all experiments reported in the paper. The following sections list the commands used to reproduce the corresponding paper results. All commands should be run directly from the repository root directory.
