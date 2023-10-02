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

## Getting Started

### Docker

If you have Docker installed just build the image:

```
docker build -t fer-ms .
```

and run a container:

```
docker run -p 8000:8000 fer-ms
```

you may change the first `8000` to map the service to the desired port on your machine.

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
