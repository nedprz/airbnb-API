import logging
import random

from fastapi import APIRouter
import pandas as pd
from pydantic import BaseModel, Field, validator

log = logging.getLogger(__name__)
router = APIRouter()


class Item(BaseModel):
    """Use this data model to parse the request body JSON."""

    
    ZipCode: int = Field(..., example=11215)
    RoomType: str = Field(..., example='Entire home/apt')
    Accomodates: int = Field(..., example=3)

    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        return pd.DataFrame([dict(self)])

    @validator('ZipCode')
    def ZipCode_must_be_positive(cls, value):
        """Validate that ZipCode is a positive number."""
        assert value > 0, f'ZipCode == {value}, must be > 0'
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
