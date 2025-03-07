# base-image/Dockerfile
FROM continuumio/miniconda3

# Make ClassifyAnything directory working dir
WORKDIR /app

# Copy the conda env creation yaml file to local
COPY conda_env.yml /app/conda_env.yml

# RUN the conda create env command to create an env from yaml file
RUN conda env create -f /app/conda_env.yml

# Set the environment activation as the default shell command.
SHELL ["conda", "run", "-n", "BaseCondaEnv", "/bin/bash", "-c"]

# Copy all necessary code to container
COPY cache /app/cache
COPY configs /app/configs
COPY classification /app/classification
COPY results /app/results
COPY test_images /app/test_images
COPY Dockerfile /app/Dockerfile
COPY main.py /app/main.py
COPY readme.md /app/readme.md

# Run main.py as entry point
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "BaseCondaEnv", "python", "/app/main.py", "-cf", "/app/configs/base.json"]