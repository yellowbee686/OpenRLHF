FROM nvcr.io/nvidia/pytorch:24.02-py3

WORKDIR /app

RUN set -eux && \
    apt-get update && \
    apt-get install -y gosu && \
    rm -rf /var/lib/apt/lists/* && \
    gosu nobody true

RUN apt-get update && apt-get -y install sudo
RUN sudo su -

RUN DEBIAN_FRONTEND=noninteractive apt install -y tzdata

RUN apt-get -y install build-essential git python3-dev python3-pip libopenexr-dev libxi-dev libglfw3-dev libglew-dev libomp-dev libxinerama-dev libxcursor-dev gdb
RUN pip uninstall xgboost transformer_engine flash_attn -y
RUN python -m pip install --upgrade pip
# 以下为vllm的依赖，可能会卡在这些安装处，因此尝试手动安装
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple vllm_nccl_cu12==2.18.1.0.4.0
# tinghua源找不到对应版本的vllm
# RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple vllm==0.4.2
# RUN pip install vllm==0.4.2
# vllm的一些依赖的hash不对齐，尝试默认trust
RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org vllm==0.4.2

COPY docker-entrypoint.sh .
RUN chmod a+x docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
