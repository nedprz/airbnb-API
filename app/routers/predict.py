import logging
import random
#from keras.models import Model
#from tensorflow import keras

#import pickle
from fastapi import APIRouter
import pandas as pd
from pydantic import BaseModel, Field, validator
from joblib import load 
from flask import Flask, jsonify, request


log = logging.getLogger(__name__)
router = APIRouter()


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

model = load('model.pkl')
result = model.predict(data_df)


@router.post('/predict')
async def predict(item: Item):
    """Make random baseline predictions for classification problem."""
    X_new = item.to_df()
    log.info(X_new)
    y_pred = random.randint(20,1000) 
    
    return {
        y_pred
    }


@app.post("/items/")
async def create_item(item: Item):
    y_pred = random.randint(100,500) 
    return y_pred


