# Import base image
FROM huggingface/transformers-pytorch-gpu

# Save files from host
COPY ./swnm /home/app

# Change working directory
WORKDIR "/home/app"

# Pip install requirements
RUN python3 -m pip install -r requirements.txt
