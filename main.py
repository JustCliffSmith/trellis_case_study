import pickle

from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel

# Load the model files outside of the inference function
# so we're not reloading every time there is an API call.
with open("./models/tfidf.pickle", "rb") as f:
    tfidf = pickle.load(f)
with open("./models/clf.pickle", "rb") as f:
    clf = pickle.load(f)


def infer_document(document: str) -> str:
    """Run inference on a single document and return the predicted class.

    Parameters
    ----------
    document : str
        Document to run inference on.

    Returns
    -------
    str
        Predicted class.
    """
    # Same TODO as from the model notebook. A pipeline would be cleaner.
    document = tfidf.transform([document])
    prediction_probability = clf.predict_proba(document)
    return prediction_probability


def evaluate_prediction_probability(
    prediction_probabilities: np.array, prediction_threshold: float = 0.155
) -> str:
    """_summary_

    Parameters
    ----------
    prediction_probabilities : np.array
        _description_
    prediction_threshold : float, optional
        _description_, by default 0.155

    Returns
    -------
    str
        Predicted class.
    """
    # If no probabilities are above the threshold.
    if sum(prediction_probabilities > prediction_threshold) == 0:
        return "other"
    else:
        # Get the index of the highest probability and return the corresponding class.
        return clf.classes_[np.argmax(prediction_probabilities)]

class Document(BaseModel):
    document_text: str


app = FastAPI(
    title="Trellis Data Science Case Study",
    description="Serves a machine learning model to predict certain classes from a document.",
    version="1.0.0",
)


@app.get("/")
async def root():
    """Give a friendly hello!"""
    return {"message": "Hello, this is a simple API for document inference."}


@app.post("/classify_document")
async def classify_document(document: Document) -> dict:
    """Classify a single text document.

    Parameters
    ----------
    document : Document
        Request body containing a single document.

    Returns
    -------
    dict
        Response object for a successful classification.

    Raises
    ------
    HTTPException
        Raised when the Request body does not have the proper field.
    HTTPException
        Raised when the document is empty.
    """
    # By default FastAPI will throw a 422 "Error: Unprocessable Entity"
    # if document_text isn't present in the body. Let's modify it to make the
    # error message more explicit.
    try:
        document = document.document_text
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Request body does not contain field `document_text`.",
        )

    # Note: trying to pass a requst body with more than one document already throws a 422.
    # TODO test that it is only one document

    if document == "":
        raise HTTPException(status_code=400, detail="Document text is empty.")

    prediction_probability = infer_document(document)[0]
    predicted_label = evaluate_prediction_probability(prediction_probability)
    return {"message": "Classification successful", "label": predicted_label}