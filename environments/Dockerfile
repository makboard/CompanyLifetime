# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.3.1-base-ubuntu20.04

# Set non-interactive shell to avoid prompts during build
ARG DEBIAN_FRONTEND=noninteractive

# Remove any third-party apt sources to avoid issues with expiring keys
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    sox \
    libsox-dev \
    libsox-fmt-all \
    build-essential \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    ffmpeg \
    libsm6 \
    libxext6 \
    unzip \
 && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user and set permissions
ARG DOCKER_NAME
ARG DOCKER_USER_ID
ARG DOCKER_GROUP_ID
RUN groupadd -g $DOCKER_GROUP_ID $DOCKER_NAME && \
    adduser --disabled-password --uid $DOCKER_USER_ID --gecos '' --shell /bin/bash $DOCKER_NAME && \
    echo "$DOCKER_NAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-$DOCKER_NAME

# Create and set the working directory
WORKDIR /app
RUN chown -R $DOCKER_NAME:$DOCKER_NAME /app

# Switch to the non-root user
USER $DOCKER_NAME
ENV HOME=/home/$DOCKER_NAME

# Set up Conda environment
ENV CONDA_AUTO_UPDATE_CONDA=false \
    PATH=$HOME/miniconda/bin:$PATH
COPY --chown=$DOCKER_NAME environment.yml /app/environment.yml
RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_23.1.0-1-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p ~/miniconda && \
    rm ~/miniconda.sh && \
    conda env create --file=environment.yml && \
    rm /app/environment.yml && \
    conda clean -ya

# Configure shell to activate Conda environment by default
ENV PATH $HOME/miniconda/envs/stat_env/bin:$PATH
RUN echo "source activate stat_env" > ~/.bashrc
SHELL ["conda", "run", "-n", "stat_env", "/bin/bash", "-c"]

# Optional: Expose any necessary ports dynamically
# EXPOSE 

# Define default command to run container
CMD ["bash"]