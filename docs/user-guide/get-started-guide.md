# Get Started Guide

-   **Time to Complete:** Less than 1 hour
-   **Programming Language:** Go, Python, HTTP REST

## Get Started

### Prerequisites

- Docker Engine* 24.0.4 or greater
- Docker Compose* 2.19.1 or greater
- cURL* or another data transfer tool compatible with an RESTful HTTP API

### Step 1: Build

Build the service:
```
docker compose build
```

### Step 2: Run

Start the service:
```
docker compose up
```
Confirm the status is live and reachable by using the `/status` endpoint with *curl* or another HTTP request tool of choice:
```
curl --location 'localhost:9001/status'
```
If the service is live, it should respond with JSON similar to below:
```
{"Models":null,"Datasets":null,"Results":null,"Tasks":null}
```

## Build a New Model with a Built-In Dataset

Using the microservice requires communicating with the RESTful HTTP API. The below commands use *cURL**, but can be adapted to your tool of choice. 

The microservice requires a dataset to train a model. This example uses the CLUSTER option, which generates a randomized dataset with detectable clusters for demonstration purposes. 

1.  Train the classifier model.
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
The `dataset` key should match your chosen dataset. The `generate_classic_model` key will allow you to generate a comparison model and will provide you with a performance difference between an unoptimzied and optimized model.

 You should see a response similar to below:
 ```
{
    "Name": "showcase_test_1",
    "ID": "m1",
    "TrainedDataset": "CLUSTER",
    "ID_num": 1,
    "OptPath": "/storage/models/showcase_test_1_QUANTIZED.model",
    "ClascPath": "/storage/models/showcase_test_1_RANDOM.model",
    "Models": [
        {
            "Type": "RANDOM",
            "AUCROC": 0.6485292318982391,
            "TrainTime": 1.0858993530273438,
            "TestTime": 0.5000574588775635,
            "AveTreeDepth": 15.370646814179207,
            "AveLeafCount": 306.83,
            "AveSplitNodeCount": 611.66,
            "AveTreeSize": 918.49
        },
        {
            "Type": "QUANTIZED",
            "AUCROC": 0.6576578298271369,
            "TrainTime": 0.808140754699707,
            "TestTime": 0.5961713790893555,
            "AveTreeDepth": 12.325323709915143,
            "AveLeafCount": 193.91,
            "AveSplitNodeCount": 385.82,
            "AveTreeSize": 579.73
        }
    ]
}
```

With the request provided above, the response contains information about optimized and unoptimized versions of the model.

2. Download the Model

You can download the model with a simple request. Note that this will return the binary representation of the model, so you should pipe this output into a file if using *cURL* or save the response in your request tool.
```
curl --location --request GET 'localhost:9001/models' \
--header 'Content-Type: text/plain' \
--data 'modelid="m1"
modeltype="QUANTIZED"'
```
The `modelid` key should match the id of your desired model.

## Performance

No performance metrics to be shared at this time.

## Summary

In this get started guide, you learned how to: 
 - Build the microservice
 - Start the microservice
 - Train a new model with a provided dataset
 - Download the model
## Learn More

-   Follow step-by-step examples to become familiar with the core
    functionality of the microservice, in
    [Tutorials](tutorials.md).
-   Understand the components, services, architecture, and data flow, in
    the [Overview](overview.md).

## Troubleshooting

The microservice runs as a Docker Compose service. Data in containers should be considered ephemeral. If the service stops working, first you should try to bring down, dispose, rebuild, and bring back up the service:
```
docker compose down -v
docker compose build
docker compose up
```

### Error Logs

The microservice runs as a Docker Compose service. You can view the logs of the container using Docker. The default name of the service is `isoforest_microservice`:
```
docker logs isoforest_microservice
```

## Known Issues

-   Uploaded datasets may show an incorrect number of datapoints in the upload response.

