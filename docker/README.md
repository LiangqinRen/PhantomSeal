# Docker Environment

This Docker image packages the PhantomSeal runtime environment. It does not include datasets, checkpoints, logs, or third-party project directories. After entering the container, run `tools/setup.sh` to initialize third-party projects and download the required data and checkpoints.

## Build

Run the build command from the repository root directory.

```shell
docker build -f docker/Dockerfile -t phantomseal:ccs2026 .
```

## Run

The host machine must have a working NVIDIA driver, Docker Engine, and NVIDIA Container Toolkit. Verify GPU access first.

```shell
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

Run PhantomSeal directly from the repository copy inside the image. Do not use `--rm` if you want downloaded datasets, checkpoints, third-party projects, and logs to remain available after exiting.

```shell
docker run -it --gpus all \
    --shm-size=32g \
    --name phantomseal \
    phantomseal:ccs2026
```

The `phantomseal` Conda environment is activated automatically when the shell starts. Verify PyTorch CUDA support inside the container.

```shell
git pull
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

To re-enter the same container later, use:

```shell
docker start -ai phantomseal
```

For development, you can instead mount a host checkout into the container. In that mode, Git may require the mounted path to be marked as safe before `git pull`.

```shell
docker run --rm -it --gpus all \
    --shm-size=32g \
    -v "$PWD":/workspace/PhantomSeal \
    -w /workspace/PhantomSeal \
    phantomseal:ccs2026

git config --global --add safe.directory /workspace/PhantomSeal
git pull
```

## Prepare Artifacts

Datasets, checkpoints, and third-party projects are prepared by the setup script.

```shell
bash tools/setup.sh
```

## Run Experiments

Use the reproduce scripts from the repository root. Without an argument, each script runs all experiments in its default list. A selector such as `1`, `1,3`, or `1,3-5` runs only selected experiments.

```shell
bash reproduce/table5.sh 1
```

## Docker Hub

After the image is validated, tag and push it to Docker Hub.

```shell
docker tag phantomseal:ccs2026 <dockerhub-user>/phantomseal:ccs2026
docker push <dockerhub-user>/phantomseal:ccs2026
```
