pip install blenderproc
# conda install -c conda-forge libstdcxx-ng=12.1
apt-get update && apt-get install -y libgl1-mesa-glx libxrender-dev libxi6 libxkbcommon-x11-0
apt-get update && apt-get upgrade -y && apt-get install -y \
    git \
    libfontconfig1 \
    libfreeimage-dev \
    libgl1 \
    libjpeg-dev \
    libxi-dev \
    libxrender1 \
    libxxf86vm1 \
    zlib1g-dev \
    libsm6 \
    libglib2.0-0 \
    libglfw3-dev \
    libgles2-mesa-dev
