import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Union

import uvicorn
from fastapi import (BackgroundTasks, Depends, FastAPI, File, HTTPException,
                     UploadFile)
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from starlette.responses import FileResponse

from ml import Model

app = FastAPI()

dirname = os.path.dirname(__file__)

db = []

model: Model = Model()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class User(BaseModel):
    email: str
    full_name: Union[str, None] = None
    disabled: Union[bool, None] = None

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user_dict = fake_users_db.get(form_data.username)
    if not user_dict:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    user = UserInDB(**user_dict)
    hashed_password = fake_hash_password(form_data.password)
    if not hashed_password == user.hashed_password:
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    return {"access_token": user.username, "token_type": "bearer"}

@app.post("/predict")
async def create_upload_file(
        background_tasks: BackgroundTasks, file: UploadFile = File()
) -> JSONResponse:
    """
    :param file:
    :param background_tasks:
    :return JSONResponse:
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
    print("-------- Most probable diseases gradcams computed --------")


@app.get("/gradcam/{uuid}/{disease}")
async def get_gradcam_for(uuid: str, disease: str) -> FileResponse:
    """
    :param uuid:
    :param disease:
    :return FileResponse:
    """
    gradcam_file = Path(f"./xrays/predict/{uuid.lower()}/{disease.lower()}.png")
    print(gradcam_file)
    if not gradcam_file.exists():
        model.generate_gradcam_unique(disease_label=disease, img_uuid=uuid)
    if gradcam_file.is_file():
        return FileResponse(gradcam_file)


@app.get("/diseases")
async def get_predictions_labels():
    diseases_json = {"diseases": model.get_labels()}
    return json.dumps(diseases_json)


@app.on_event("shutdown")
def clean_predictions():
    predict_path = "./xrays/predict"
    for uuid_dir in os.listdir(predict_path):
        path = os.path.join(predict_path, uuid_dir)
        try:
            shutil.rmtree(path)
            print("-------- Past predictions removed --------")
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (path, e))
    

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=80, workers=2, reload=True)
