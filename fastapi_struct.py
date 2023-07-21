import io
import pandas as pd

from fastapi import FastAPI, Path, Request
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
import rating_prediction2
import rating_prediction
from starlette.exceptions import HTTPException

from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline


app = FastAPI()

# Step 1: Parse JSON file and load into memory
import json

with open("alexa_reviews.json") as f:
    data = json.load(f)


class Reviews:
    id: int
    review: str
    rating: int

    def __init__(self, id, review, rating):
        self.id = id
        self.review = review
        self.rating = rating



class newReviews(BaseModel):
    id: int
    review: str = Field(min_length= 3, max_length=200)
    rating: int = Field(gt=0, lt=6)

    class Config:
        json_schema_extra = {
            'example': {
                'id': 'optional',
                'review': 'I love interacting with alexa',
                'rating': 5
            }
        }


class requestReview(BaseModel):
    new_rev: str


@app.post("/new_review_predict_with_pipeline/add_rev/")
async def create_rev(new_rev: requestReview):
    if not new_rev.new_rev:
        raise HTTPException(status_code=422, detail="Empty review text")

    review_list = [new_rev.new_rev]
    # Call the prediction function with the DataFrame
    predictions = rating_prediction.pipeline.predict(review_list)

    # Assuming the prediction function returns a list of predictions
    prediction = predictions[0]
    return int(prediction)




@app.post("/new_review_predict/add_rev/")
async def create_rev(new_rev: requestReview):
    if not new_rev.new_rev:
        raise HTTPException(status_code=422, detail="Empty review text")

    review_list = [new_rev.new_rev]
    bow_review = rating_prediction2.bow_transformer.transform(rating_prediction2.text_process(review_list))
    tfidf_review = rating_prediction2.tfidf_transformer.transform(bow_review)

    prediction = rating_prediction2.rating_model.predict(tfidf_review)[0]
    # Return the prediction as a response
    return int(prediction)



#return all the data
@app.get("/reviews/")
async def get_reviews():
    return data


#search by id number across reviews
@app.get("/reviews_by_id/{review_id}")
async def get_review_by_id(review_id: str):
    if review_id not in data:
        raise HTTPException(status_code=404, detail="Review not found")
    return data[review_id]



















