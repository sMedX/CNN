FROM nvidia/cuda:8.0-cudnn6-devel

WORKDIR /app

ADD . /app

# Install Python3
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Install any needed packages specified in requirements.txt
RUN apt-get update -y && \
    apt-get install -y libgtk2.0-dev && \
    apt-get install -y cmake && \
    apt-get install -y python3-tk && \   
    pip3 install --trusted-host pypi.python.org -r docker/requirements.txt

# Set AGG as default backend for matplotlib
RUN sed -i.bak 's/: tkagg/: agg/g' /usr/local/lib/python3.5/dist-packages/matplotlib/mpl-data/matplotlibrc

# Make ports 8888 8890 available to the world outside this container 
EXPOSE 8888 8890

CMD ["python3", "./train.py", "-c", "./data/config.json", "-r", "./data/models/snapshot_epoch_115"]
