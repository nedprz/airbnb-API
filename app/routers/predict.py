import logging
import random

from fastapi import APIRouter
import pandas as pd
from pydantic import BaseModel, Field, validator

log = logging.getLogger(__name__)
router = APIRouter()


class Item(BaseModel):
    """Use this data model to parse the request body JSON."""

    SquareFootage: float = Field(..., example=3000)
    ZipCode: int = Field(..., example=11215)
    String: str = Field(..., example='yes')

    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        return pd.DataFrame([dict(self)])

    @validator('SquareFootage')
    def SquareFootage_must_be_positive(cls, value):
        """Validate that SquareFootage is a positive number."""
        assert value > 0, f'SquareFootage == {value}, must be > 0'
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
