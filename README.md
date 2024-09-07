# FER-Microservice

This repository contains a simple webservice
offering an HTTP API to perform
Facial Expression Recognition (FER).

The employed FER model is specialized on facial images in which the upper half of the face is occluded. 

POST a single image containing a human face
and get a response with probabilities for seven basic emotions:

```bash
curl -X 'POST' \
  'http://localhost:8000/recognize/file' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@fer/test/test_image.png'
```

```json

{
  "anger": 0.051,
  "disgust": 0.049,
  "fear": 0.1,
  "happiness": 0.7,
  "neutral": 0.099,
  "sadness": 0.001,
  "surprise": 0.0
}
```

## Getting Started

### Model Files
You will need a dedicated GPU and have (CUDA) drivers installed for the models to work.   
You need to download three model files from Google Drive:  
1. Download [ir50.pth](https://drive.google.com/file/d/17QAIPlpZUwkQzOTNiu-gUFLTqAxS-qHt/view?usp=sharing)
and put it in the [pretrain](fer/posterv2/pretrain) directory.  
2. Download [mobilefacenet_model_best.pth.tar](https://drive.google.com/file/d/1SMYP5NDkmDE3eLlciN7Z4px-bvFEuHEX/view?usp=sharing)
and put it in the [pretrain](fer/posterv2/pretrain) directory.  
3. Download [affectnet-7-model_best_state_dict_only.pth](https://drive.google.com/file/d/10NWqIcEAHjScAGlCKryEpWgiKJvyVlaF/view?usp=sharing)
and put it in the [posterv2](fer/posterv2) directory.  

The first two model links reference the same models used in the
[POSTER_V2 repository](https://github.com/Talented-Q/POSTER_V2?tab=readme-ov-file#preparation).
The Poster_V2 or Poster++ architecture was developed by Mao et al. in:
```
@article{mao2023poster,
  title={POSTER V2: A simpler and stronger facial expression recognition network},
  author={Mao, Jiawei and Xu, Rui and Yin, Xuesong and Chang, Yuanqi and Nie, Binling and Huang, Aibin},
  journal={arXiv preprint arXiv:2301.12149},
  year={2023},
  doi={10.48550/arXiv.2301.12149}
}
```
Their code is published on GitHub under the MIT License:
[POSTER_V2 repository](https://github.com/Talented-Q/POSTER_V2?tab=readme-ov-file#preparation).
We trained the third model to download using their code (with small adjustments) and trained with images which we had
occluded artificially.


### Docker

If you have Docker installed just build the image:

```
docker build -t fer-ms .
```

and run a container:

```
docker run --gpus all -p 8000:8000 fer-ms
```

You may change the first `8000` to map the service to the desired port on your machine.

### Python

You may find a complete exemplary machine/environment definition in this repository's [Dockerfile](./Dockerfile).

If you just want to run the application,
install all application requirements by running:

```
pip install -r requirements.txt
```

and then run the application via:

```
uvicorn fer.main:app --host 0.0.0.0 --port 8000
```

If you also want to run the tests,
install all requirements by running:

```
pip install -r requirements-dev.txt
```

and then, to run the test:

```
python -m pytest fer/test
```

## Example Usage

You can find the API definition in [main.py](./fer/main.py).

### curl

To GET a ping:

```bash
curl -X 'GET' \
  'http://localhost:8000/ping' \
  -H 'accept: application/json'

```

To POST an image file:

```bash
curl -X 'POST' \
  'http://localhost:8000/recognize/file' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@fer/test/test_image.png'
```

To POST a base64 encoded image:

```bash
curl -X 'POST' \
  'http://localhost:8000/recognize/base64' \
  -H 'accept: application/json' \
  -H 'Content-Type: text/plain' \
  -d '@fer/test/test_image_base64.txt'
```

### Tests in Python

Look around the [fer/test](./fer/test) folder for examples in Python code.
