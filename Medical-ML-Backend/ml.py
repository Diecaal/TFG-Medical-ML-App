import pandas as pd
import tensorflow as tf

import model_creation
from gradcam import compute_gradcam, generate_gradcam, generate_gradcam_unique

labels = [
    "Cardiomegaly",
    "Emphysema",
    "Effusion",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Atelectasis",
    "Pneumothorax",
    "Pleural_Thickening",
    "Pneumonia",
    "Fibrosis",
    "Edema",
    "Consolidation",
]

# (KEY) [uuid: str]
# (VALUES) predictions: np_arr[0][x], preprocessed_input: np_arr[][]
dict_preprocessed_inputs = {}


class Model:
    def __init__(self) -> None:
        self.IMAGE_DIR = "./xrays/computed"
        self.PREDICTIONS_IMAGE_DIR = "./xrays/predict"
        self.df = pd.read_csv("./csv/train-small.csv")
        self.model, self.graph, self.session = model_creation.get_model()

    def get_labels(self):
        return labels

    def predict_for_image(self, img_uuid: str):
        json, predictions, preprocessed_input = compute_gradcam(
            model=self.model,
            img_uuid=img_uuid,
            predict_image_dir=self.PREDICTIONS_IMAGE_DIR,
            images_dir=self.IMAGE_DIR,
            df=self.df,
            labels=labels,
        )

        dict_preprocessed_inputs[img_uuid] = predictions, preprocessed_input

        return json

    def generate_gradcam_initial(self, img_uuid: str):
        print("--------- Starting generation of gradcam images ---------")
        generate_gradcam(
            model=self.model,
            img_uuid=img_uuid,
            images_dir=self.IMAGE_DIR,
            predict_image_dir=self.PREDICTIONS_IMAGE_DIR,
            df=self.df,
            labels=labels,
            selected_labels=labels,
            graph=self.graph,
            session=self.session,
            predictions=dict_preprocessed_inputs[img_uuid][0],
            preprocessed_input=dict_preprocessed_inputs[img_uuid][1],
        )

    def generate_gradcam_unique(self, img_uuid: str, disease_label: str):
        print("--------- Starting generation of gradcam images ---------")
        generate_gradcam_unique(
            model=self.model,
            img_uuid=img_uuid,
            images_dir=self.IMAGE_DIR,
            predict_image_dir=self.PREDICTIONS_IMAGE_DIR,
            df=self.df,
            labels=labels,
            selected_labels=[disease_label],
            predictions=dict_preprocessed_inputs[img_uuid][0],
            preprocessed_input=dict_preprocessed_inputs[img_uuid][1],
        )
