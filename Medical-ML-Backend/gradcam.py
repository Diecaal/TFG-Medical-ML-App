import random
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.utils.image_utils import load_img
from pathlib import Path
import json

tf.compat.v1.disable_eager_execution()
# from tensorflow.compat.v1.logging import INFO, set_verbosity
# from tensorflow.compat.v1 import disable_eager_execution


def get_mean_std_per_batch(image_dir, df, H=320, W=320):
    sample_data = []
    for img in df.sample(100)["Image"].values:
        image_path = os.path.join(image_dir, img)
        sample_data.append(np.array(load_img(image_path, target_size=(H, W))))

    mean = np.mean(sample_data, axis=(0, 1, 2, 3))
    std = np.std(sample_data, axis=(0, 1, 2, 3), ddof=1)
    return mean, std


def load_image(img, predict_image_dir, image_dir, df, preprocess=True, H=320, W=320):
    """Load and preprocess image."""
    mean, std = get_mean_std_per_batch(image_dir, df, H=H, W=W)
    img_path = os.path.join(predict_image_dir, img)
    x = load_img(img_path, target_size=(H, W))
    if preprocess:
        x -= mean
        x /= std
        x = np.expand_dims(x, axis=0)
    return x


def grad_cam(input_model, image, cls, layer_name, H=320, W=320):
    """GradCAM method for visualizing input saliency."""
    y_c = input_model.output[0, cls]
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]

    gradient_function = K.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # Process CAM
    cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)  # type: ignore
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam


def build_json(img_uuid, predictions, labels):
    image_predictions = {"uuid": img_uuid, "predictions": []}
    for i in range(len(predictions[0])):
        image_predictions["predictions"].append(
            {"disease": labels[i], "prediction": float(predictions[0][i])}
        )

    return json.dumps(image_predictions)


def compute_gradcam(
    model, img_uuid: str, predict_image_dir: str, images_dir: str, df, labels
):
    preprocessed_input = load_image(
        f"{img_uuid}/original.png", predict_image_dir, images_dir, df
    )
    predictions = model.predict(preprocessed_input)

    return build_json(img_uuid, predictions, labels), predictions, preprocessed_input


def generate_gradcam(
    model,
    img_uuid: str,
    predict_image_dir: str,
    images_dir: str,
    df,
    labels,
    selected_labels,
    predictions,
    preprocessed_input,
    layer_name="bn",
):
    if not os.path.exists(f"./plots/{img_uuid}/"):
        os.makedirs(f"./plots/{img_uuid}/")

    print("Loading original image")
    plt.figure(figsize=(15, 10))
    plt.title("Original")
    plt.axis("off")
    plt.imshow(
        np.array(
            load_image(
                f"{img_uuid}/original.png",
                predict_image_dir,
                images_dir,
                df,
                preprocess=False,
            )
        ),
        cmap="gray",
    )
    plt.savefig(f"./plots/{img_uuid}/original")

    for i in range(len(labels)):
        if float(predictions[0][i]) < 0.6:
            continue
        if labels[i] in selected_labels:
            print(f"Generating gradcam for class {labels[i]}")
            gradcam = grad_cam(model, preprocessed_input, i, layer_name)
            img_plot = plt.figure(figsize=(15, 10))
            plt.title(f"{labels[i]}: p={predictions[0][i]:.3f}")
            plt.axis("off")
            plt.imshow(
                np.array(
                    load_image(
                        f"{img_uuid}/original.png",
                        predict_image_dir,
                        images_dir,
                        df,
                        preprocess=False,
                    )
                ),
                cmap="gray",
            )
            plt.imshow(gradcam, cmap="jet", alpha=min(0.5, predictions[0][i]))
            plt.savefig(f"./xrays/predict/{img_uuid}/{labels[i]}")
