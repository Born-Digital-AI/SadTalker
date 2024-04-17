FROM nvidia/cuda:12.4.1-base-ubuntu20.04

ENV PYTHON_VERSION=3.10

WORKDIR /opt

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y wget ffmpeg libgl1 git python3.10 python3.10-dev python3.10-venv python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


RUN mkdir ./checkpoints
RUN wget -nc https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar -O  ./checkpoints/mapping_00109-model.pth.tar
RUN wget -nc https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar -O  ./checkpoints/mapping_00229-model.pth.tar
RUN wget -nc https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors -O  ./checkpoints/SadTalker_V0.0.2_256.safetensors
RUN wget -nc https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors -O  ./checkpoints/SadTalker_V0.0.2_512.safetensors

RUN mkdir -p ./gfpgan/weights
RUN wget -nc https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth -O ./gfpgan/weights/alignment_WFLW_4HG.pth
RUN wget -nc https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth -O ./gfpgan/weights/detection_Resnet50_Final.pth
RUN wget -nc https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -O ./gfpgan/weights/GFPGANv1.4.pth
RUN wget -nc https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth -O ./gfpgan/weights/parsing_parsenet.pth


ENV PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt /opt/
ENV VIRTUAL_ENV=/opt/.venv
RUN python3.10 -m venv $VIRTUAL_ENV
ENV PATH "$VIRTUAL_ENV/bin:$PATH"

RUN python3 -m pip install --upgrade pip && pip install --no-cache-dir --upgrade -r /opt/requirements.txt

RUN pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

COPY /examples /opt/examples
COPY /src /opt/src
COPY *.py *.png /opt/

ENV RUN_ENV=docker

CMD ["python3", "app.py"]