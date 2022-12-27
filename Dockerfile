# Import base image
FROM huggingface/transformers-pytorch-gpu

# Save files from host
COPY ./swnm /home/app
