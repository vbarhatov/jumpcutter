#!/usr/bin/env bash

HOST=$1

ssh "$HOST" 'apt-get update && apt-get install -y ffmpeg && apt update && apt install -y python3.7 python3-pip && pip3 install Pillow audiotsm scipy numpy pytube3 quote extract'

scp ./jumpcutter.py "$HOST:~/"
scp ./jumpcutter.sh "$HOST:~/"
ssh "$HOST" 'sudo chmod +x ~/jumpcutter.sh'

# enable swap to avoid OOM but I don't recommend relying on it as it makes the processing extremely slow
ssh "$HOST" "sudo swapoff -a && sudo dd if=/dev/zero of=/swapfile bs=1G count=2 && sudo mkswap /swapfile && sudo swapon /swapfile"

