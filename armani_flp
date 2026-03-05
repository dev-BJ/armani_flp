#!/bin/sh

echo "APT update"
apt update

echo "APT full-upgrade"
apt full-upgrade -y

echo "Install Essentials"
apt install git build-essential cmake ffmpeg -y

echo "Download TensorRT deb"
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/local_repos/nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-12.0_1.0-1_amd64.d
eb

echo "Install TensorRT Repo"
dpkg -i nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-12.0_1.0-1_amd64.deb

echo "Set TensorRT keyring"
cp /var/nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-12.0/*-keyring.gpg /usr/share/keyrings/

echo "Update after TensorRT"
apt update

echo "Install TensorRT(8.6.1) and deps"
apt-get install tensorrt=8.6.1.6-1+cuda12.0 libnvinfer8=8.6.1.6-1+cuda12.0 libnvinfer-dev=8.6.1.6-1+cuda12.0 libnvinfer-plugin8=8.6.1.6-1+cuda12.0 libnvinfer-bin=8.6.1.6-1+cuda12.0 libnvinfer-lean-dev=8.6.1.6-1+cuda12.0 \
	libnvinfer-plugin-dev=8.6.1.6-1+cuda12.0 libnvinfer-vc-plugin-dev=8.6.1.6-1+cuda12.0 libnvinfer-dispatch-dev=8.6.1.6-1+cuda12.0 libnvonnxparsers-dev=8.6.1.6-1+cuda12.0 libnvinfer-samples=8.6.1.6-1+cuda12.0 \
	libnvinfer-headers-dev=8.6.1.6-1+cuda12.0 libnvinfer-headers-dev=8.6.1.6-1+cuda12.0 libnvinfer-headers-dev=8.6.1.6-1+cuda12.0 libnvinfer-headers-plugin-dev=8.6.1.6-1+cuda12.0 \
	libnvinfer-headers-plugin-dev=8.6.1.6-1+cuda12.0 -y

echo "Python TensorRT Package"
apt-get install python3-libnvinfer-dev=8.6.1.6-1+cuda12.0 python3-libnvinfer=8.6.1.6-1+cuda12.0 python3-libnvinfer-lean=8.6.1.6-1+cuda12.0 python3-libnvinfer-dispatch=8.6.1.6-1+cuda12.0 -y

echo "Build Grid Sample"
export PATH=/usr/local/cuda/bin:$PATH
cd armani_lp/flp/grid-sample3d-trt-plugin
rm -rf build && mkdir build && cd build
cmake .. -DTensorRT_ROOT=/usr/lib/python3.10/dist-packages/tensorrt
make

echo "Back to workspace"
cd /workspace
rm -rf nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-12.0_1.0-1_amd64.deb
