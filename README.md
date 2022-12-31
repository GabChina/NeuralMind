# SWNM - Sliding Window Pipeline
This repository started to keep track of the NM Sliding Window Pipeline for training the main BERT model.

## Connect to DGX
1. Connect using SSH:
```sh
ssh [user]@[public-ip] -p [ssh-port]
```

## Usage
1. Build the image:
```sh
sudo docker build -t swnm:latest . --force-rm --no-cache
```

2. Run the container in interactive/detached mode:
```sh
sudo docker run -it -d \
--name="aluno_gabrieldamata-neuralmind" \
--rm \
--cpus="8.0" \
--gpus device=5 \
--memory="64g" \
-v /raid/gabriel_da_mata/neuralmind/swnm:/home/app/checkpoints \
swnm
```

3. Enter exec mode using the container ID
```sh
docker ps --filter ancestor=swnm

docker exec -it [container-id] /bin/bash
```

4. Login to wandb
```sh
wandb login [wandb-key] --relogin
```

5. Download the dataset
```sh
gdown [dataset-gdown-link]
```

6. Run main script
```sh
python3 -m main
```

Note that currently you can't close the ssh connection before the script finishes.
