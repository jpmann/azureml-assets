#PTCA image
FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu117-py38-torch1131:{{latest-image-tag}}

# ls root dir
RUN ls -lh

USER root
RUN apt-get -y update

# pip install requirements
# ADD wheels wheels
# RUN pip install wheels/*
RUN pip install azureml-acft-accelerator
RUN pip install azureml-acft-contrib-hf-diffusion

# Dependencies for using HF-Flavour mlflow
RUN pip install azureml-evaluate-mlflow

RUN pip install \
        datasets==2.9.0 \
        mpi4py==3.1.3 \
        accelerate==0.16.0 \
        optimum==1.6.4 \
        transformers==4.26.1 \
        diffusers==0.12.0 \
        onnxruntime-training==1.14.1
        # torchvision==0.14.1+cu117     # handled by ACPT image

# Dependecies for model deployment
RUN pip install azure-ai-ml==1.0.0

# Check versions
RUN pip list
