from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from starlette.responses import FileResponse
import json
from pydantic import BaseModel
from ml import Model
import uvicorn
import os
import uuid
import shutil

app = FastAPI()

dirname = os.path.dirname(__file__)

db = []

model: Model = Model()


@app.post("/predict")
async def create_upload_file(file: UploadFile = File()):  # -> JSONResponse:
    """
    :param file:
    :return img_uuid:
    """
    print("-------- File received --------")

    if not os.path.exists("xrays/"):
        os.mkdir("xrays/")

    if not os.path.exists("xrays/predict"):
        os.mkdir("xrays/predict")

    img_uuid = str(uuid.uuid4())
    os.mkdir(f"xrays/predict/{img_uuid}")

    with open(f"xrays/predict/{img_uuid}/original.png", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        json_content = jsonable_encoder(model.predict_for_image(f"{img_uuid}"))
        print("-------- Predictions returned --------")
        print(json_content)
        return JSONResponse(content=json_content)


@app.get("/diseases")
async def get_predictions_labels():
    diseases_json = {"diseases": model.get_labels()}
    print(json.dumps(diseases_json))
    return json.dumps(diseases_json)


if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=80, workers=2, reload=True)  #
