FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
      python3-pip python3-dev git ninja-build wget \
      ffmpeg libsm6 libxext6 \
      openmpi-bin libopenmpi-dev && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /usr/src/app
COPY . .

ENV FORCE_CUDA=1

# Install PyTorch with CUDA 11.8 compatibility
RUN python -m pip install --upgrade pip && \
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Patch make.sh for V100S compilation
RUN cd ops && \
    bash make.sh && \
    cd ..

# Install remaining dependencies
RUN pip install git+https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once.git@33f2c898fdc8d7c95dda014a4b9ebe4e413dbb2b \
    && pip install git+https://github.com/facebookresearch/segment-anything.git \
    && pip install git+https://github.com/UX-Decoder/Semantic-SAM.git@package \
    && pip install mpi4py==3.1.4 \
    && pip install openai==0.28.1 \
    && pip install gradio==4.43.0

# Download pretrained models
RUN sh download_ckpt.sh

EXPOSE 6092
ENV GRADIO_SERVER_NAME="0.0.0.0"

#RUN chmod +x /usr/src/app/entrypoint.sh
#CMD ["/usr/src/app/entrypoint.sh"]

CMD ["tail", "-f", "/dev/null"]