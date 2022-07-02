#syntax=docker/dockerfile:1

FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
COPY environment.yaml environment.yaml
RUN conda env create -f environment.yaml

# Activate environment
SHELL ["conda", "run", "-n", "asi-final-project", "/bin/bash", "-c"]

COPY . .
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "asi-final-project", "python", "src/application.py"]