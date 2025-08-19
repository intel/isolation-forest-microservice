# Optimized Isolation Forest Classifier Training and Inference Microservice 
The Optimized Isolation Forest Classifier: Training and Inference Microservice, or ISOForest, is an isolation forest classifier creation and inference tool designed to use new Intel-desigend optimization techniques to create smaller, faster, and still as accurate isolation forest models for classification and regression tasks.

ISOForest is delivered as a flexible microservice, capable of being used either on its own through its RESTful HTTP API or integrated into a wider microservice-based system.
## Running as a Docker Container
The application is built to run as a Docker container. This application is built and tested with Docker Engine 24.0.4 and Docker Compose v2.19.1.

```
docker compose build
docker compose up
```

You can use the "CLUSTER" key for generating a random set. For using your own datasets, use the "CUSTOM" value and providing the dataset ID.

## Sample API Commands
### Status
```
curl --location 'localhost:9001/status'
```
### Upload Dataset
```
curl --location 'localhost:9001/data/upload' \
--form 'file=@"/<full path to>/custom.csv"'
```
### Train Model
```
curl --location 'localhost:9001/train' \
--header 'Content-Type: text/plain' \
--data 'title = "ISO Forest Classifier Configuration"

name = "showcase_test_1"
task = "showcase"
dataset = "CLUSTER"
compare = true
generate_optimized_model = true
generate_classic_model = true
saved_model_name = "showcase_test_1"
path = "model_QUANTIZED.model"'
```
### Infer with Model
```
curl --location 'localhost:9001/infer' \
--header 'Content-Type: text/plain' \
--data 'datasetid="d1"
modelid="m1"'
```
### Download Model
```
curl --location --request GET 'localhost:9001/models' \
--header 'Content-Type: text/plain' \
--data 'modelid="m1"'
```

### Get Model Tree
```
curl --location --request GET 'localhost:9001/models/tree' \
--header 'Content-Type: text/plain' \
--data 'modelid="m1"'
```
