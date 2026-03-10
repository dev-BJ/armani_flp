FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    git curl ca-certificates build-essential\
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python \
 && ln -sf /usr/bin/python3.10 /usr/bin/python3

RUN python -m pip install --upgrade pip wheel setuptools

RUN git clone --depth 1 https://github.com/dev-BJ/armani_flp /opt/armani_flp
WORKDIR /opt/armani_flp/pl

RUN pip install --no-cache-dir -r requirements_base.txt

# Ensure the folder exists, then download weights into ./pretrained_weights
RUN mkdir -p ./pretrained_weights && mkdir -p ./pretrained_weights/onnx 
 && RUN mkdir -p ./pretrained_weights/personalive && mkdir -p ./pretrained_weights/tensorrt
RUN python tools/download_weights.py

RUN pip install --no-cache-dir -r requirements_trt.txt

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PORT=80
EXPOSE 80
CMD ["/entrypoint.sh"]
