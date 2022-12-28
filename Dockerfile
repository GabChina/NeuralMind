# Cuda version 10.2.89
FROM huggingface/transformers-pytorch-gpu:4.9.1

# Save files from host
COPY ./swnm /home/app

# Change working directory
WORKDIR "/home/app"

# Pip install requirements
RUN python3 -m pip install -r requirements.txt
