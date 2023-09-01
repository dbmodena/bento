FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:openjdk-r/ppa

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
RUN ln -fs /usr/share/zoneinfo/Europe/Rome /etc/localtime
RUN dpkg-reconfigure -f noninteractive tzdata

RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    python3-openssl \
    git

RUN wget https://www.python.org/ftp/python/3.9.2/Python-3.9.2.tgz \
    && tar xvf Python-3.9.2.tgz

RUN cd Python-3.9.2 \
    && ./configure --enable-optimizations \
    && make -j 8 \
    && make altinstall

WORKDIR /app
ADD ./requirements.txt /app
RUN python3.9 -m pip install --upgrade pip
RUN pip install -r requirements.txt --use-pep517
RUN python3.9 -m pip install -U memory_profiler
ENTRYPOINT ["python3.9", "-u", "run_algorithm.py"]