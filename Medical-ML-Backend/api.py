from fastapi import FastAPI, File, UploadFile, Request, BackgroundTasks
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
from pathlib import Path

app = FastAPI()

dirname = os.path.dirname(__file__)

db = []

model: Model = Model()


@app.post("/predict")
async def create_upload_file(
    background_tasks: BackgroundTasks, file: UploadFile = File()
) -> JSONResponse:
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

        # Computation of more likely diseases will be compute in the background
        background_tasks.add_task(initial_gradcams, img_uuid)

        return JSONResponse(content=json_content)


def initial_gradcams(img_uuid: str):
    model.generate_gradcam_initial(img_uuid=img_uuid)


@app.get("/gradcam/{uuid}/{disease}")
async def get_gradcam_for(uuid: str, disease: str) -> FileResponse | None:
    gradcam_file = Path(f"./xrays/predict/{uuid.lower()}/{disease.lower()}.png")
    if not gradcam_file.exists():
        model.generate_gradcam_unique(disease_label=disease, img_uuid=uuid)
    if gradcam_file.is_file():
        return FileResponse(gradcam_file)
    return None


@app.get("/diseases")
async def get_predictions_labels():
    diseases_json = {"diseases": model.get_labels()}
    return json.dumps(diseases_json)


if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=80, workers=2, reload=True)
