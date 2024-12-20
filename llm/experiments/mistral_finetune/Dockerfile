FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV PYTHONUNBUFFERED=1 
ENV PIP_DISABLE_PIP_VERSION_CHECK=on 
# ENV PIP_DEFAULT_TIMEOUT=100 
ENV POETRY_VERSION=1.7.1 
ENV POETRY_HOME="/opt/poetry" 
ENV POETRY_NO_INTERACTION=1 
ENV POETRY_VIRTUALENVS_CREATE=false 
ENV VENV_PATH="/app/.venv" 
ENV CUDA_HOME=${CUDA_HOME}/usr/local/cuda-12.1
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-12.1/lib64
ENV BNB_CUDA_VERSION=121
    # Install system tools and libraries.
    # Utilize --mount flag of Docker Buildx to cache downloaded packages, avoiding repeated downloads

RUN --mount=type=cache,id=apt-build,target=/var/cache/apt \
    apt-get update && \ 
    apt-get install -y software-properties-common && \
    # Add the Deadsnakes PPA for Python 3.11
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        git \
        wget \
        ca-certificates \
        curl \
        build-essential \
        python3.11 \
        python3-pip \
        python3.11-venv \
        python3.11-dev && \
    wget https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1-Linux-x86_64.sh -O /tmp/cmake-install.sh && \
    chmod +x /tmp/cmake-install.sh && \
    /tmp/cmake-install.sh --skip-license --prefix=/usr/local && \
    rm /tmp/cmake-install.sh && \
    # Clean up to keep the image size small
    rm -rf /var/lib/apt/lists/*  && \
    # Set Python 3.11 as the default Python version
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --set python3 /usr/bin/python3.11 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --set python /usr/bin/python3.11

# Set PATH to include Poetry and custom venv
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python - --version $POETRY_VERSION

# Create and prepare the virtual environment
RUN python -m venv $VENV_PATH && \
    python -m pip install --upgrade pip && \
    pip cache purge && rm -Rf /root/.cache/pip/http
    
WORKDIR /app
COPY . .
# Copy dependency files to the app directory
COPY poetry.lock pyproject.toml /app/

# Install dependencies with Poetry and Torch with pip, caching downloaded packages
RUN --mount=type=cache,target=/root/.cache \
    poetry install && \
    # pip install torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116 && \
    pip install torch==2.1.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121 && \
    git clone https://github.com/TimDettmers/bitsandbytes.git && \
    cd bitsandbytes && \
    pip install -r requirements-dev.txt && \
    cmake -DCOMPUTE_BACKEND=cuda -S .  && \
    make && \
    pip install . && \
    ln -s /app/.venv/lib/python3.11/site-packages/bitsandbytes/libbitsandbytes_cuda121.so /app/.venv/lib/python3.11/site-packages/bitsandbytes/libbitsandbytes_cuda121_nocublaslt121.so


# Copy the entire project code to the container
COPY ./ /source/

CMD ["python", "mistral_finetune.py"]