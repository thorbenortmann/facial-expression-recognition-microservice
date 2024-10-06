FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3.9 \
    python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ./fer /app/fer

# Download models from Google Drive if not locally available
RUN if [ ! -f /app/fer/posterv2/pretrain/ir50.pth ] || \
       [ ! -f /app/fer/posterv2/pretrain/mobilefacenet_model_best.pth.tar ] || \
       [ ! -f /app/fer/posterv2/affectnet-7-model_best_state_dict_only.pth ]; then \
        pip3 install gdown==4.7.3 && \
        ( [ ! -f /app/fer/posterv2/pretrain/ir50.pth ] && \
          gdown https://drive.google.com/uc?id=12QjP4CJOzNn7j4AkfNfODXpwRpu5O9ZT -O /app/fer/posterv2/pretrain/ir50.pth ) && \
        ( [ ! -f /app/fer/posterv2/pretrain/mobilefacenet_model_best.pth.tar ] && \
          gdown https://drive.google.com/uc?id=1oMGcEyhqQLjaca4pppEYV8xONmXU6ucv -O /app/fer/posterv2/pretrain/mobilefacenet_model_best.pth.tar ) && \
        ( [ ! -f /app/fer/posterv2/affectnet-7-model_best_state_dict_only.pth ] && \
          gdown https://drive.google.com/uc?id=10NWqIcEAHjScAGlCKryEpWgiKJvyVlaF -O /app/fer/posterv2/affectnet-7-model_best_state_dict_only.pth ); \
    fi

COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "fer.main:app", "--host", "0.0.0.0", "--port", "8000"]
