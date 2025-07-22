from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Define a request body model
class InputData(BaseModel):
    text: str

@app.post("/reply")
def reply(data: InputData):
    if data.text.lower() == "hello":
        return {"response": "world"}
    else:
        return {"response": "I don't understand"}
