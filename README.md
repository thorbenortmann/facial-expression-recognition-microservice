# FER-Microservice

This repository contains a simple webservice
offering an HTTP API to perform
Facial Expression Recognition (FER).

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

## Facial Expression Recognition under Partial Occlusion
This branch of the repository contains the setup we used in the study described in our paper
[EmojiHeroVR: A Study on Facial Expression Recognition under Partial Occlusion from Head-Mounted Displays](https://doi.org/10.48550/arXiv.2410.03331).
The models used, 
[11-07-11-59-model_best_state_dict_only.pth](https://drive.google.com/file/d/1Niww_GBbVBq2nx9ZCuHOe75FT2grgZaX/view?usp=drive_link) 
and [11-10-09-22-model_best_state_dict_only.pth](https://drive.google.com/file/d/1Uuf3jUVeRdM24WKavIUyWg6ztlYUyjwV/view?usp=drive_link),
were trained on artificially occluded images from the
[AffectNet](https://doi.org/10.1109/TAFFC.2017.2740923),
[EmotioNet](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Benitez-Quiroz_EmotioNet_An_Accurate_CVPR_2016_paper.html),
[ExpW](https://doi.org/10.1007/s11263-017-1055-1),
[FER+](https://doi.org/10.1145/2993148.2993165),
[SFEW](https://doi.org/10.1109/ICCVW.2011.6130508) and
[KDEF](https://doi.org/10.1080/02699930701626582)
datasets. 

## Getting Started

### Model Files

You will need a dedicated GPU and have (CUDA) drivers installed for the models to work.   
You need to download three model files from Google Drive:

1. Download [ir50.pth](https://drive.google.com/file/d/17QAIPlpZUwkQzOTNiu-gUFLTqAxS-qHt/view?usp=sharing)
and put it in the [pretrain](fer/posterv2/pretrain) directory.  
2. Download [mobilefacenet_model_best.pth.tar](https://drive.google.com/file/d/1SMYP5NDkmDE3eLlciN7Z4px-bvFEuHEX/view?usp=sharing)
and put it in the [pretrain](fer/posterv2/pretrain) directory.  
3. Download [11-07-11-59-model_best_state_dict_only.pth](https://drive.google.com/file/d/1Niww_GBbVBq2nx9ZCuHOe75FT2grgZaX/view?usp=drive_link) 
and put it in the [posterv2](fer/posterv2) directory.
4. Download [11-10-09-22-model_best_state_dict_only.pth](https://drive.google.com/file/d/1Uuf3jUVeRdM24WKavIUyWg6ztlYUyjwV/view?usp=drive_link) 
and put it in the [posterv2](fer/posterv2) directory.

#### Attribution

The models,
[ir50.pth](https://drive.google.com/file/d/17QAIPlpZUwkQzOTNiu-gUFLTqAxS-qHt/view?usp=sharing) and
[mobilefacenet_model_best.pth.tar](https://drive.google.com/file/d/1SMYP5NDkmDE3eLlciN7Z4px-bvFEuHEX/view?usp=sharing),
used in this branch of the repository originate from [Talented-Q](https://github.com/Talented-Q)'s
[POSTER_V2 repository](https://github.com/Talented-Q/POSTER_V2/tree/18de5591c3fa0b7b22bb9fe2d61e7f813e6e3b08).
Their code is published on GitHub under the
[MIT License](https://github.com/Talented-Q/POSTER_V2/blob/18de5591c3fa0b7b22bb9fe2d61e7f813e6e3b08/LICENSE).
The [corresponding publication](https://doi.org/10.48550/arXiv.2301.12149) is:

```
@article{mao2023poster,
  title={POSTER V2: A simpler and stronger facial expression recognition network},
  author={Mao, Jiawei and Xu, Rui and Yin, Xuesong and Chang, Yuanqi and Nie, Binling and Huang, Aibin},
  journal={arXiv preprint arXiv:2301.12149},
  year={2023},
  doi={10.48550/arXiv.2301.12149}
}
```
The models,
[11-07-11-59-model_best_state_dict_only.pth](https://drive.google.com/file/d/1Niww_GBbVBq2nx9ZCuHOe75FT2grgZaX/view?usp=drive_link) and
[11-10-09-22-model_best_state_dict_only.pth](https://drive.google.com/file/d/1Uuf3jUVeRdM24WKavIUyWg6ztlYUyjwV/view?usp=drive_link),
were trained by us using the training code of [Talented-Q](https://github.com/Talented-Q)'s
[POSTER_V2 repository](https://github.com/Talented-Q/POSTER_V2/tree/18de5591c3fa0b7b22bb9fe2d61e7f813e6e3b08).

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

**Note:** You will need the
[Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)
to run containers with GPU support.
We tested this on a host machine with:
- Ubuntu: `22.04`
- Nvidia Driver Version: `535.183.01`
- CUDA Version: `12.2`
- cuDNN Version: `8.9.5`
- Docker Version: `27.3.1`
- Nvidia Container Toolkit-Version: `1.16.2`
- GPU: `Nvidia GeForce RTX 3090`

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

### API via curl

You can find the API definition in [main.py](./fer/main.py).

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

### FER Live Demo

With the virtual environment activated, execute:

```bash
python demo.py
```

### Tests in Python

Look around the [fer/test](./fer/test) folder for examples in Python code.
