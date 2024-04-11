# First stage: Download the model files
FROM python:3.10-slim-bullseye as model_downloader

WORKDIR /opt

RUN apt-get update && apt-get install -y wget

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

# Second stage: Build the actual application container with the model files
FROM python:3.10-slim-bullseye

WORKDIR /opt
RUN apt-get update && apt-get install -y ffmpeg libgl1 git

COPY --from=model_downloader /opt/checkpoints /opt/checkpoints
COPY --from=model_downloader /opt/gfpgan /opt/gfpgan

COPY requirements.txt /opt/
RUN python -m pip install --upgrade pip && pip install -r /opt/requirements.txt

COPY /examples /opt/examples
COPY /src /opt/src
COPY *.py *.png /opt/

ENV RUN_ENV=docker

CMD ["python", "app.py"]