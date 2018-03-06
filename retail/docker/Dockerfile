FROM nvidia/cuda:8.0-cudnn6-devel

WORKDIR /app

# Install Python3
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Install Git and clone darknet
RUN apt-get update -y && \
    apt-get install -y git
RUN git clone https://github.com/pjreddie/darknet

# Install OpenCV
RUN apt-get install -y libopencv-dev python-opencv && \
    pip3 install opencv-python

# Configure and build darknet
RUN cd darknet && \
    echo '1,3c1,3\n< GPU=0\n< CUDNN=0\n< OPENCV=0\n---\n> GPU=1\n> CUDNN=1\n> OPENCV=1' > Makefile.patch && \
    diff patch Makefile Makefile.patch && \
    make
