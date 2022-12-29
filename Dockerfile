# Cuda version 10.2.89
FROM huggingface/transformers-pytorch-gpu:4.9.1

# Save files from host
COPY ./swnm /home/app

# Change working directory
WORKDIR "/home/app"

# I wish I was dead
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update && apt-get upgrade -y

RUN apt install software-properties-common -y && add-apt-repository ppa:deadsnakes/ppa -y && apt-get update -y

RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

# Install python3.9
RUN apt-get install python3.9 -y

#Verifies python version
RUN python3 --version && python3.9 --version

# Upgrade pip
RUN python3.9 -m pip install --upgrade pip

#RUN export  PATH="$PATH:/usr/lib/python3.9"
RUN rm -f /usr/lib/python && ln -s /usr/lib/python /usr/lib/python3
# Pip install requirements
RUN python3.9 -m pip install -r requirements.txt
