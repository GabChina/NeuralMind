# Downloading base image from transformers
FROM huggingface/transformers-pytorch-gpu

# Save files from host
COPY ./swnm /home/app

# Change working directory
WORKDIR "/home/app"

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Pip install requirements
RUN python3 -m pip install -r requirements.txt
