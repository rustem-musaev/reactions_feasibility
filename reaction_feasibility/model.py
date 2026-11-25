"""Multilayer perceptron, custom metrics and data generators."""
import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization, Dense, Input
from tensorflow.keras.metrics import Precision, Recall, SpecificityAtSensitivity
from tensorflow.keras.models import Model


# --- Metrics (computed per batch; need run_eagerly=True) ---------------------

def recall(y_true, y_pred):
    m = Recall()
    m.update_state(y_true, y_pred)
    return m.result().numpy()


def f1(y_true, y_pred):
    m = Precision()
    m.update_state(y_true, y_pred)
    precision = m.result().numpy()

    m = Recall()
    m.update_state(y_true, y_pred)
    rec = m.result().numpy()

    return 2 * ((precision * rec) / (precision + rec + K.epsilon()))


def balanced_acc(y_true, y_pred):
    """Mean of sensitivity and specificity.

    Specificity is taken at the threshold giving the batch's sensitivity.
    """
    sensitivity = recall(y_true, y_pred)
    specificity = SpecificityAtSensitivity(sensitivity)
    specificity.update_state(y_true, y_pred)
    specificity = specificity.result().numpy()
    return (sensitivity + specificity) / 2


# --- Model -------------------------------------------------------------------

def build_mlp(n_descriptors, hidden_units=1000):
    """BatchNorm -> Dense(hidden_units, relu) -> BatchNorm -> Dense(1, sigmoid)."""
    model_input = Input(shape=(n_descriptors,))
    x = BatchNormalization()(model_input)
    x = Dense(hidden_units, activation="relu")(x)
    x = BatchNormalization()(x)
    output = Dense(1, activation="sigmoid", name="output_t")(x)
    model = Model(inputs=model_input, outputs=output)

    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=[f1, balanced_acc, keras.metrics.AUC(name="auc")],
                  run_eagerly=True)
    model.summary()
    return model


def n_descriptors_in(shelf):
    """Number of descriptor columns, read from the first stored entry."""
    first_key = next(iter(shelf.keys()))
    return shelf[first_key][0].shape[1]


# --- Data --------------------------------------------------------------------

def batch_generator(keys, batch_size, shelf, ratios=None):
    """Endlessly yield (descriptors, labels) batches for `keys` from an open shelve.

    If a list is passed as `ratios`, the share of decoys in each batch is
    appended to it (used to check class balance).
    """
    number_of_batches = len(keys) / batch_size
    counter = 0
    while True:
        batch_keys = keys[batch_size * counter:batch_size * (counter + 1)]
        x_batch = np.stack([shelf[k][0].values[0] for k in batch_keys])
        y_batch = np.stack([shelf[k][1] for k in batch_keys])
        if ratios is not None:
            ratios.append(sum(y_batch) / len(y_batch))
        counter += 1
        yield x_batch, y_batch
        if counter >= number_of_batches:
            counter = 0
