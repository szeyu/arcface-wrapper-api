# ArcFace Face Vector API

## Requirements

- Docker + Docker Compose

## Local development

```bash
# install dependencies
make install
# download ONNX models to arcface-api/models
make models
# start Postgres only
make db
# run the API locally
make run
```

## Start with Docker Compose

```bash
docker-compose up --build
```

* API runs on `http://localhost:3000/api`
* Postgres runs on `localhost:5432` (DB: face_db)
* Required ONNX models (ArcFace + SCRFD) are downloaded automatically during the Docker build

## Endpoints

* `POST /api/store_embedding` → { image_base64 }
* `POST /api/compare` → { image_base64_A, image_base64_B }
* `POST /api/search` → { image_base64, top_k }
* `GET /api/image/:id`
* `GET /api/list?limit=N`
* `DELETE /api/item/:id`
