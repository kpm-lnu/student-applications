# docker build -t advection-diffusion . 

FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
  wget \
  && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O miniconda.sh \
  && bash miniconda.sh -b -p /opt/conda \
  && rm miniconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN conda create -n fenics -c conda-forge fenics-dolfin python=3.9 -y

SHELL ["conda", "run", "-n", "fenics", "/bin/bash", "-c"]

WORKDIR /app

COPY advection_diffusion.py .

RUN pip install numpy matplotlib pandas scipy