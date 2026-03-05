#!/bin/bash

set -e  # Exit on any error
set -u  # Treat unset variables as errors

echo "============== APT update =============="
apt-get update

echo

echo "============== APT full-upgrade =============="
apt-get full-upgrade -y

echo

echo "============== Install Essentials =============="
apt-get install git build-essential cmake ffmpeg -y

echo

echo "============== Upgrade pip essentials =============="
python -m pip install --upgrade pip wheel setuptools

echo

echo "============== Download TensorRT .deb =============="
rm -rf *.deb
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/local_repos/nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-12.0_1.0-1_amd64.deb

echo

echo "============== Install TensorRT Repo =============="
dpkg -i nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-12.0_1.0-1_amd64.deb

echo

echo "============== Set TensorRT keyring =============="
cp /var/nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-12.0/*-keyring.gpg /usr/share/keyrings/
echo "-------- TensorRT keyring copied ---------"

echo

echo "============== Update after TensorRT =============="
apt-get update

echo

echo "============== Install TensorRT(8.6.1) and deps =============="
apt-get install tensorrt=8.6.1.6-1+cuda12.0 libnvinfer8=8.6.1.6-1+cuda12.0 libnvinfer-dev=8.6.1.6-1+cuda12.0 libnvinfer-plugin8=8.6.1.6-1+cuda12.0 libnvinfer-bin=8.6.1.6-1+cuda12.0 libnvinfer-lean-dev=8.6.1.6-1+cuda12.0 \
	libnvinfer-plugin-dev=8.6.1.6-1+cuda12.0 libnvinfer-vc-plugin-dev=8.6.1.6-1+cuda12.0 libnvinfer-dispatch-dev=8.6.1.6-1+cuda12.0 libnvonnxparsers-dev=8.6.1.6-1+cuda12.0 libnvinfer-samples=8.6.1.6-1+cuda12.0 \
	libnvinfer-headers-dev=8.6.1.6-1+cuda12.0 libnvinfer-headers-dev=8.6.1.6-1+cuda12.0 libnvinfer-headers-dev=8.6.1.6-1+cuda12.0 libnvinfer-headers-plugin-dev=8.6.1.6-1+cuda12.0 \
	libnvinfer-headers-plugin-dev=8.6.1.6-1+cuda12.0 -y

apt-mark hold tensorrt=8.6.1.6-1+cuda12.0 libnvinfer8=8.6.1.6-1+cuda12.0 libnvinfer-dev=8.6.1.6-1+cuda12.0 libnvinfer-plugin8=8.6.1.6-1+cuda12.0 libnvinfer-bin=8.6.1.6-1+cuda12.0 libnvinfer-lean-dev=8.6.1.6-1+cuda12.0 \
	libnvinfer-plugin-dev=8.6.1.6-1+cuda12.0 libnvinfer-vc-plugin-dev=8.6.1.6-1+cuda12.0 libnvinfer-dispatch-dev=8.6.1.6-1+cuda12.0 libnvonnxparsers-dev=8.6.1.6-1+cuda12.0 libnvinfer-samples=8.6.1.6-1+cuda12.0 \
	libnvinfer-headers-dev=8.6.1.6-1+cuda12.0 libnvinfer-headers-dev=8.6.1.6-1+cuda12.0 libnvinfer-headers-dev=8.6.1.6-1+cuda12.0 libnvinfer-headers-plugin-dev=8.6.1.6-1+cuda12.0 \
	libnvinfer-headers-plugin-dev=8.6.1.6-1+cuda12.0

echo

echo "============== Python TensorRT Package =============="
apt-get install python3-libnvinfer-dev=8.6.1.6-1+cuda12.0 python3-libnvinfer=8.6.1.6-1+cuda12.0 python3-libnvinfer-lean=8.6.1.6-1+cuda12.0 python3-libnvinfer-dispatch=8.6.1.6-1+cuda12.0 -y
apt-mark hold python3-libnvinfer-dev=8.6.1.6-1+cuda12.0 python3-libnvinfer=8.6.1.6-1+cuda12.0 python3-libnvinfer-lean=8.6.1.6-1+cuda12.0 python3-libnvinfer-dispatch=8.6.1.6-1+cuda12.0

echo

echo "============== Build Grid Sample =============="
#export PATH=/usr/local/cuda/bin:$PATH
#cd /workspace/armani_lp/flp/grid-sample3d-trt-plugin
#rm -rf build && mkdir build && cd build
#cmake .. -DTensorRT_ROOT=/usr/lib/python3.10/dist-packages/tensorrt
#make

echo

echo "============== Setup Python Environment =============="
#cd /workspace/armani_lp
#rm -rf /workspace/armani_lp/.venv
#python -m venv .venv --system-site-packages
#. /workspace/armani_lp/.venv/bin/activate
#echo "---------- Python Evironment Set ----------"

echo

echo "============== Install Python requirements =============="
#python -m pip install --upgrade pip wheel setuptools
#pip install -r requirements.txt
# pip install huggingface_hub[cli]

echo

echo "============== Download ONNX model files =============="
#cd flp
#hf download warmshao/FasterLivePortrait --local-dir ./checkpoints

echo

echo "============== Convert ONNX to TRT =============="
#sh scripts/all_onnx2trt.sh

echo

echo "============== Clean up =============="
rm -rf /workspace/nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-12.0_1.0-1_amd64.deb
