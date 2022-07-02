FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu18.04

CMD ["bash"]

RUN rm /bin/sh && ln -s /bin/bash /bin/sh # buildkit

RUN apt-get update

RUN apt-get install -y curl

RUN apt-get install -y git

RUN apt-get install -y wget

RUN apt-get install -y unzip

RUN apt-get update

RUN apt-get install -y python-pip

RUN apt-get install -y libgl1-mesa-dev

RUN curl https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh -o Miniconda3-py39_4.10.3-Linux-x86_64.sh # buildkit

RUN bash Miniconda3-py39_4.10.3-Linux-x86_64.sh -b # buildkit

ENV PATH=/root/miniconda3/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN conda init bash # buildkit

RUN conda update -y conda # buildkit

RUN conda create --name diffusion python=3.9 # buildkit

RUN echo "source activate diffusion" > ~/.bashrc # buildkit

RUN git clone https://github.com/discus0434/disco-diffusion.git

WORKDIR /disco-diffusion

RUN chmod +x setup.sh

RUN ./setup.sh
