SHELL=/bin/bash

BASE=$(shell pwd)

ACTIVATE := $(BASE)/venv/bin/activate

python := $(BASE)/venv/bin/python3

#-- GPU parametrisation --
export CUDA_VISIBLE_DEVICES=0
# CUDA_LAUNCH_BLOCKING=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64

pip_install:
	$(python) -m pip install --upgrade pip setuptools testresources wheel
	$(python) -m pip install --no-cache-dir -r requirements.txt

jpt:
	source ${ACTIVATE} && \
	jupyter lab --port=5004 --ip=0.0.0.0 ${BASE}

check_port:
	sudo lsof -n -i :5000 | grep LISTEN

check_gpu:
	$(python) utils.py

qdrant:
	docker run --privileged -p 6333:6333 qdrant/qdrant

redis:
	docker run --rm -p 6379:6379 redis/redis-stack-server:latest