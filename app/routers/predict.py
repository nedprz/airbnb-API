import logging
import random
from keras.models import Model
from tensorflow import keras

import pickle
from fastapi import APIRouter
import pandas as pd
from pydantic import BaseModel, Field, validator
from joblib import load 
from flask import Flask, jsonify, request


log = logging.getLogger(__name__)
router = APIRouter()
model = load('model.pkl')

class Item(BaseModel):
    """Use this data model to parse the request body JSON."""

    Neighborhood: str = Field(..., example='Park Slope')
    RoomType: str = Field(..., example='Entire home/apt')
    MinimumNights: int = Field(..., example = 5)
    Accomodates: int = Field(..., example=3)

    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        return pd.DataFrame([dict(self)])

    @validator('MinimumNights')
    def MinimumNights_must_be_positive(cls, value):
        """Validate that MinimumNights is a positive number."""
        assert value > 0, f'MinimumNights == {value}, must be > 0'
        return value


@router.post('/predict')
async def predict(item: Item):
    """Make random baseline predictions for classification problem."""
    X_new = item.to_df()
    log.info(X_new)
    y_pred = random.randint(20,1000) 
    
    return {
        y_pred
    }



@router.post('/predict2')
def predict2():
    # get data
    data = request.get_json(force=True)
    print(data)

    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)

    print(data_df.shape)
    # predictions
    result = model.predict(data_df)

    # send back to browser
    output = {'results': int(result[0])}

    # return data
    return jsonify(results=output)



