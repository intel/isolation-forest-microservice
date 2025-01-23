# Tutorials

## Tutorial 1: Build and Download a Model

In this tutorial, you will learn how to upload a dataset, build a model, and download that model.

### Time to Complete
15 minutes

### Learning Objectives

-   By the end of this tutorial, you will be able to create a new model using the microservice

### Prerequisites

You should follow the Build and Run steps in the [Getting Started Guide](get-started-guide.md) before running any tutorial.

Using the microservice requires communicating with the RESTful HTTP API. The below commands use *cURL**, but can be adapted to your tool of choice. 

### Build the Model

The microservice requires a dataset to train a model. 

1.  Train the classifier model. The below request is specific to a dataset called SAT_KURT:
```
curl --location 'localhost:9001/train' \
--header 'Content-Type: text/plain' \
--data 'title = "ISO Forest Classifier Configuration"

name = "showcase_test_1"
task = "showcase"
dataset = "SAT_KURT"
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
    "TrainedDataset": "SAT_KURT",
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

### Summary

In this tutorial, you learned how to:
 - Build the microservice
 - Start the microservice
 - Train a new model with a ready dataset
 - Download the model

## Learn More

-   Understand the architecture in
    the [Overview](overview.md).

## Troubleshooting

The microservice runs as a Docker Compose service. Data in containers should be considered ephemeral. If the service stops working, first you should try to bring down, dispose, rebuild, and bring back up the service:
```
docker compose down -v
docker compose build
docker compose up
```

### Error Logs

The microservice runs as a Docker Compose service. You can view the logs of the container using Docker:
```
docker logs isoforest_microservice
```

## Known Issues

