FROM pytorch/torchserve:latest-cpu

COPY . /home/model-server/
WORKDIR /home/model-server/

RUN mkdir -p model-store && \
    torch-model-archiver \
        --model-name test-model \
        --version 1.0 \
        --serialized-file test_model.pt \
        --handler handler.py \
        --extra-files "question2idx.json,user2idx.json" \
        --export-path ./model-store

CMD ["torchserve", \
     "--start", \
     "--ts-config=config.properties", \
     "--model-store=model-store", \
     "--models=test-model.mar"]