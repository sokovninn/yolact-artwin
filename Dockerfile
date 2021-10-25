FROM continuumio/miniconda3

COPY environment.yml .
RUN apt-get -y update && \
    apt-get install -y gcc libc-dev ffmpeg libsm6 libxext6 && \
    conda env create -f environment.yml && \
    git clone https://github.com/sokovninn/yolact-artwin.git

# ENTRYPOINT exec bash -c "source activate /opt/conda/envs/yolact-env/ && cd /yolact-artwin"

# sudo docker build -t yolact-artwin .
# sudo docker run -t -d --rm --gpus=all --mount src=/local/datagen/yolact-artwin/data,target=/yolact-artwin/data,type=bind yolact-artwin
# sudo docker exec -it 7a7124ca8555 bash -c "cd yolact-artwin && conda run -n yolact-env python eval.py --trained_model=data/weights/yolact_base_54_800000.pth --image=data/yolact_example_0.png:data/rest.png"
# sudo docker exec -it c960f03cf892 bash
