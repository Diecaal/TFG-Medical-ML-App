from gradcam import compute_gradcam, generate_gradcam
import pandas as pd
import model_creation

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


class Model:
    def __init__(self) -> None:
        self.IMAGE_DIR = "./xrays/computed"
        self.PREDICTIONS_IMAGE_DIR = "./xrays/predict"
        self.df = pd.read_csv("./csv/train-small.csv")
        self.eval_model()

    def eval_model(self):
        self.model = model_creation.get_model()

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

        print("--------- Starting generation of gradcam images ---------")

        generate_gradcam(
            model=self.model,
            img_uuid=img_uuid,
            images_dir=self.IMAGE_DIR,
            predict_image_dir=self.PREDICTIONS_IMAGE_DIR,
            df=self.df,
            labels=labels,
            selected_labels=labels,
            predictions=predictions,
            preprocessed_input=preprocessed_input,
        )

        return json
