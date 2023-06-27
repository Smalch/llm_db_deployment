FROM mosaicml/composer:latest



RUN pip3.10 install --upgrade pip


RUN pip install chromadb
RUN pip install langchain
RUN pip install fastapi
RUN pip install transformers
RUN pip install accelerate
RUN pip install openai
#RUN pip install xformers
RUN pip install pypdf
RUN pip install PyPDF2
RUN pip install sentence_transformers
RUN pip install InstructorEmbedding
RUN pip install einops
RUN pip install uvicorn
RUN pip install pydantic
RUN pip install python-multipart
RUN pip install diffusers
RUN pip install duckduckgo-search
RUN pip install --upgrade torch
RUN pip install -qU transformers accelerate einops langchain wikipedia xformers
RUN pip install -qU triton
#RUN pip install triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python
WORKDIR /
COPY . .

#
#RUN pip install -r requirements.txt

EXPOSE 8000
#ENTRYPOINT ["uvicorn", "API_model:app", "--host", "0.0.0.0"]




















#FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
#
## Set up time zone.
#ENV PYTHONUNBUFFERED=1
#
## SYSTEM
#RUN apt-get update --yes --quiet && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
#    software-properties-common \
#    build-essential apt-utils \
#    wget curl vim git ca-certificates kmod \
#    nvidia-driver-525 \
# && rm -rf /var/lib/apt/lists/*
#
## PYTHON 3.10
#RUN add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update --yes --quiet
#RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
#    python3.10 \
#    python3.10-dev \
#    python3.10-distutils \
#    python3.10-lib2to3 \
#    python3.10-gdbm \
#    python3.10-tk \
#    pip
#
#RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 999 \
#    && update-alternatives --config python3 && ln -s /usr/bin/python3 /usr/bin/python
#
#WORKDIR /
#COPY . .
#
#RUN pip install --upgrade pip
#
## ANACONDA
#RUN wget -O /tmp/anaconda.sh https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh \
#    && bash /tmp/anaconda.sh -b -p /anaconda \
#    && eval "$(/anaconda/bin/conda shell.bash hook)" \
#    && conda init \
#    && conda update -n base -c defaults conda \
#    && conda create --name env \
#    && conda activate env \
#    && conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch-nightly -c nvidia
#
#SHELL ["conda", "run", "--no-capture-output", "-n", "env", "pip", "install", "-r", "srequirements.txt"]
#
#
##RUN pip install -r requirements.txt
#EXPOSE 8000
#ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "env", "uvicorn", "API_model:app", "--host", "0.0.0.0"]