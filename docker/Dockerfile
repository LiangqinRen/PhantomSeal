FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    wget \
    git \
    git-lfs \
    build-essential \
    cmake \
    pkg-config \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniconda.sh \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda clean -afy

WORKDIR /workspace/PhantomSeal

COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml \
    && conda clean -afy \
    && conda init bash \
    && echo "conda activate phantomseal" >> /root/.bashrc

COPY . /workspace/PhantomSeal

RUN git remote set-url origin https://github.com/LiangqinRen/PhantomSeal.git || true

RUN conda run -n phantomseal python -V \
    && conda run -n phantomseal python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

CMD ["/bin/bash", "-l"]
