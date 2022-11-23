import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K
from sklearn.metrics import roc_auc_score, roc_curve

from keras.models import load_model

import tensorflow as tf

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

train_df = pd.read_csv("./csv/train-small.csv")
valid_df = pd.read_csv("./csv/valid-small.csv")
test_df = pd.read_csv("./csv/test.csv")


def get_train_generator(
    df,
    image_dir,
    x_col,
    y_cols,
    shuffle=True,
    batch_size=8,
    seed=1,
    target_w=320,
    target_h=320,
):
    """
    Return generator for training set, normalizing using batch
    statistics.

    Args:
      train_df (dataframe): dataframe specifying training data.
      image_dir (str): directory where image files are held.
      x_col (str): name of column in df that holds filenames.
      y_cols (list): list of strings that hold y labels for images.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.

    Returns:
        train_generator (DataFrameIterator): iterator over training set
    """
    print("getting train generator...")
    # normalize images
    image_generator = ImageDataGenerator(
        samplewise_center=True, samplewise_std_normalization=True
    )

    # flow from directory with specified batch size
    # and target image size
    generator = image_generator.flow_from_dataframe(
        dataframe=df,
        directory=image_dir,
        x_col=x_col,
        y_col=y_cols,
        class_mode="raw",
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        target_size=(target_w, target_h),
    )

    return generator


def get_test_and_valid_generator(
    valid_df,
    test_df,
    train_df,
    image_dir,
    x_col,
    y_cols,
    sample_size=100,
    batch_size=8,
    seed=1,
    target_w=320,
    target_h=320,
):
    """
    Return generator for validation set and test set using
    normalization statistics from training set.

    Args:
      valid_df (dataframe): dataframe specifying validation data.
      test_df (dataframe): dataframe specifying test data.
      train_df (dataframe): dataframe specifying training data.
      image_dir (str): directory where image files are held.
      x_col (str): name of column in df that holds filenames.
      y_cols (list): list of strings that hold y labels for images.
      sample_size (int): size of sample to use for normalization statistics.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.

    Returns:
        test_generator (DataFrameIterator) and valid_generator: iterators over test set and validation set respectively
    """
    print("getting train and valid generators...")
    # get generator to sample dataset
    raw_train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=train_df,
        directory=image_dir,
        x_col="Image",
        y_col=labels,
        class_mode="raw",
        batch_size=sample_size,
        shuffle=True,
        target_size=(target_w, target_h),
    )

    # get data sample
    batch = raw_train_generator.next()
    data_sample = batch[0]

    # use sample to fit mean and std for test set generator
    image_generator = ImageDataGenerator(
        featurewise_center=True, featurewise_std_normalization=True
    )

    # fit generator to sample from training data
    image_generator.fit(data_sample)

    # get test generator
    valid_generator = image_generator.flow_from_dataframe(
        dataframe=valid_df,
        directory=image_dir,
        x_col=x_col,
        y_col=y_cols,
        class_mode="raw",
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        target_size=(target_w, target_h),
    )

    test_generator = image_generator.flow_from_dataframe(
        dataframe=test_df,
        directory=image_dir,
        x_col=x_col,
        y_col=y_cols,
        class_mode="raw",
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        target_size=(target_w, target_h),
    )
    return valid_generator, test_generator


IMAGE_DIR = "./xrays/computed"
train_generator = get_train_generator(train_df, IMAGE_DIR, "Image", labels)
valid_generator, test_generator = get_test_and_valid_generator(
    valid_df, test_df, train_df, IMAGE_DIR, "Image", labels
)


def compute_class_freqs(labels):
    """
    Compute positive and negative frequences for each class.

    Args:
        labels (np.array): matrix of labels, size (num_examples, num_classes)
    Returns:
        positive_frequencies (np.array): array of positive frequences for each
                                         class, size (num_classes)
        negative_frequencies (np.array): array of negative frequences for each
                                         class, size (num_classes)
    """
    # total number of patients (rows)
    N = labels.shape[0]

    positive_frequencies = np.sum(labels == 1, axis=0) / N
    negative_frequencies = np.sum(labels == 0, axis=0) / N

    return positive_frequencies, negative_frequencies


freq_pos, freq_neg = compute_class_freqs(train_generator.labels)

data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": freq_pos})
data = data.append(
    [
        {"Class": labels[l], "Label": "Negative", "Value": v}
        for l, v in enumerate(freq_neg)
    ],
    ignore_index=True,
)
plt.xticks(rotation=90)
f = sns.barplot(x="Class", y="Value", hue="Label", data=data)

pos_weights = freq_neg
neg_weights = freq_pos
pos_contribution = freq_pos * pos_weights
neg_contribution = freq_neg * neg_weights

data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": pos_contribution})
data = data.append(
    [
        {"Class": labels[l], "Label": "Negative", "Value": v}
        for l, v in enumerate(neg_contribution)
    ],
    ignore_index=True,
)
plt.xticks(rotation=90)
sns.barplot(x="Class", y="Value", hue="Label", data=data)


def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    """
    Return weighted loss function given negative weights and positive weights.

    Args:
      pos_weights (np.array): array of positive weights for each class, size (num_classes)
      neg_weights (np.array): array of negative weights for each class, size (num_classes)

    Returns:
      weighted_loss (function): weighted loss function
    """

    def weighted_loss(y_true, y_pred):
        """
        Return weighted loss value.

        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
        Returns:
            loss (float): overall scalar loss summed across all classes
        """
        # initialize loss to zero
        loss = 0.0

        for i in range(len(pos_weights)):
            # for each class, add average weighted loss for that class
            loss += -1 * K.mean(
                (pos_weights[i] * y_true[:, i] * K.log(y_pred[:, i] + epsilon))
                + (
                    neg_weights[i]
                    * (1 - y_true[:, i])
                    * K.log(1 - y_pred[:, i] + epsilon)
                )
            )  # complete this line
        return loss

    return weighted_loss


def get_roc_curve(labels, predicted_vals, generator):
    auc_roc_vals = []
    for i in range(len(labels)):
        try:
            gt = generator.labels[:, i]
            pred = predicted_vals[:, i]
            auc_roc = roc_auc_score(gt, pred)
            auc_roc_vals.append(auc_roc)
            fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
            plt.figure(1, figsize=(10, 10))
            plt.plot([0, 1], [0, 1], "k--")
            plt.plot(
                fpr_rf, tpr_rf, label=labels[i] + " (" + str(round(auc_roc, 3)) + ")"
            )
            plt.xlabel("False positive rate")
            plt.ylabel("True positive rate")
            plt.title("ROC curve")
            plt.legend(loc="best")
        except:
            print(
                f"Error in generating ROC curve for {labels[i]}. "
                f"Dataset lacks enough examples."
            )
    plt.show()
    return auc_roc_vals


# predicted_vals = model.predict_generator(test_generator, steps=len(test_generator))

# auc_rocs = get_roc_curve(labels, predicted_vals, test_generator)


def get_model():
    # print("--------- Model compilation started ---------")

    # base_model = DenseNet121(weights="./models/densenet.hdf5", include_top=False)

    # x = base_model.output

    # # add a global spatial average pooling layer
    # x = GlobalAveragePooling2D()(x)

    # # and a logistic layer
    # predictions = Dense(len(labels), activation="sigmoid")(x)

    # model = Model(inputs=base_model.input, outputs=predictions)
    # model.compile(optimizer="adam", loss=get_weighted_loss(pos_weights, neg_weights))

    # # model.summary()
    # model.save("compiled_model.h5")

    # model.load_weights("./models/trained_model.h5")

    graph = tf.compat.v1.get_default_graph()
    session = tf.compat.v1.keras.backend.get_session()
    init = tf.compat.v1.global_variables_initializer()
    session.run(init)

    model = tf.keras.models.load_model("./models/full_model.h5", compile=False)

    print("--------- Model compilation finished ---------")

    return model, graph, session
