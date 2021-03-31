# Machine learning seed in Python

A seed project for starting machine learning projects, written in python.

## Requirements
  - python3

## Setup

Setup the env and dependencies:
  - `python3 -m venv env`
  - `source ./env/bin/activate`
  - `pip install -r requirements.txt`

## Usage

### Building the model

To build the model, run `./create-new-model.sh`. By default, this will save
the model in the /models/ directory.

The network variables can be edited here: [create_new_model.py](https://github.com/chrisberry4545/machine-learning-python-seed/blob/master/src/create_new_model.py).

### Serving the model

Once the model is built, the model can be served using the docker-compose file. This uses
tensorflow/serving to serve the model. It can be run using `docker-compose up`.

Once the model is running inputs can be sent to the model to get a prediction. For the
test data set included with this project you could get a prediction with the following
request:

```
curl --location --request POST 'http://localhost:8501/v1/models/model:predict' \
--header 'Content-Type: application/json' \
--data-raw '{
    "instances": [{
        "longitude":[-114.31],
        "latitude":[34.19],
        "housing_median_age":[15],
        "total_rooms":[5612],
        "total_bedrooms":[1283],
        "population":[1015],
        "households":[472],
        "median_income":[1.4936]
    }]
}'
```

