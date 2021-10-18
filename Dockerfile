FROM continuumio/miniconda3

COPY environment.yml .
RUN apt-get -y update && \
    apt-get install -y gcc libc-dev ffmpeg libsm6 libxext6 && \
    conda env create -f environment.yml && \
    git clone https://github.com/sokovninn/yolact-artwin.git

ENTRYPOINT exec bash -c "conda activate yolact-artwin && cd /yolact-artwin"

# sudo docker build -t yolact-artwin .
# sudo docker run -t -d --rm --gpus=all --mount src=/home/nikita/yolact-artwin/data,target=/yolact-artwin/data,type=bind da90aa4d5666
# sudo docker exec -it c960f03cf892 bash
