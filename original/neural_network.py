# Transcribed from NEURAL_NETWORK_CODE.pdf (printed 06/12/2020).
# Later version of the training code. Kept as it was used;
# see reaction_feasibility/model.py and scripts/train.py for the maintained version.
# Note: imports (numpy, pandas, keras, CompressShelve, metrics, callbacks,
# train_test_split, time) and `new_generator` (see ml.py) were defined
# elsewhere in the original notebook/session.

def recall(y_true, y_pred):
    m = Recall()
    m.update_state(y_true, y_pred)
    pre_res = m.result()
    res = pre_res.numpy()
    return res

def f1(y_true, y_pred):
    m = Precision()
    m.update_state(y_true, y_pred)
    pre_precision = m.result()
    precision = pre_precision.numpy()

    m = Recall()
    m.update_state(y_true, y_pred)
    pre_recall = m.result()
    recall = pre_recall.numpy()

    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def balanced_acc(y_true, y_pred):

    selectivity = recall(y_true, y_pred)
    specificity = SpecificityAtSensitivity(selectivity)
    specificity.update_state(y_true, y_pred)
    specificity = specificity.result().numpy()
    return (selectivity+specificity)/2

def encoding_mlp_generator(hidden_layers):

    import pickle
    import time
    import math
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Layer
    from tensorflow.keras.layers import Input, Dense, Dropout, concatenate, dot,BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
    from keras.models import Sequential

    with CompressShelve("test.shelve", flag="r") as f:
        desc = pd.DataFrame(f['42'][0])
    bnorm1 = BatchNormalization(trainable=True)
    bnorm2 = BatchNormalization(trainable=True)
    model_input = Input(shape=len(desc.iloc[0]))
    connect_to = bnorm1(model_input)
    hidden_layer = Dense(hidden_layers, activation="relu")(connect_to)
    connect_to = bnorm2(hidden_layer)
    print(connect_to.shape)
    output_t = Dense(1, activation="sigmoid", name="output_t")(connect_to)
    model = Model(inputs=model_input, outputs=output_t)



    model.compile(optimizer="adam", loss='binary_crossentropy', metrics = [f1,balanced_acc, keras.metrics.AUC(name='auc')], run_eagerly=True)
    model.summary()
    return model


def generate_descriptors(train):
    for x in train:
        with CompressShelve("test.shelve", flag="r") as f:
            yield f[str(x)][0]

def generate_desc_and_type(validation_set, batch_size):
    with CompressShelve("test.shelve", flag="r") as f:
        descriptors = np.array([f[str(x)][0] for n,x in enumerate(validation_set) if n < batch_size])
        labels = np.array([f[str(x)][1] for n,x in enumerate(validation_set) if n < batch_size])
        while True:
            for i in range(batch_size):
                yield descriptors, labels


def generator(validation_set, batch_size,f):
        samples_per_epoch = np.array(validation_set).shape[0]
        number_of_batches = samples_per_epoch/batch_size
        counter=0
        while True:
            x_data = validation_set[batch_size*counter:batch_size*(counter+1)]
            x_batch = np.stack([f[x][0].values[0] for x in x_data])
            y_batch = np.stack([f[x][1] for x in x_data])
            counter += 1
            yield x_batch, y_batch
            if counter >= number_of_batches:
                counter = 0




def create_model(fragmentor):

    from sklearn.linear_model import SGDClassifier
    from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
    from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
    from tensorflow.keras.callbacks import TensorBoard
    import random
    import datetime
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    with CompressShelve("test.shelve", flag="r") as f:
        indices = [u for u in f.keys()]


    indices.sort()
    x_graph = []
    y_graph = []
    pretrain, test = train_test_split(indices, test_size=0.1, random_state=42)

    train, validation_set = train_test_split(pretrain, test_size=0.2, random_state=41)
    random.shuffle(train)
    callback = EarlyStopping(monitor='loss', patience=3)
    tensorboard_callback = TensorBoard(log_dir="logs", histogram_freq=1)
    model = encoding_mlp_generator(1000)
    model.fit(new_generator(train, batch_size =200), validation_data = new_generator(validation_set, batch_size =200), verbose = 1, epochs = 100, callbacks = [callback, tensorboard_callback])


    prediction,f1_score,bac,auc = model.predict(generate_descriptors(test), batch_size = 200)
    model.save("./test_model_9111.h5")
    return model, prediction, f1_score, bac, auc



NAME = "Model - 1,678,066 Descriptors{}".format(int(time.time()))

tensorboard_callback = TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=1)

with CompressShelve("test.shelve", flag="r") as f:
    indices = list(f.keys())
mcp = ModelCheckpoint('best_model_9000.h5', save_best_only = True, monitor = 'val_balanced_acc',mode='min')
rlr = ReduceLROnPlateau(monitor='val_balanced_acc', factor=0.5, patience=1, min_lr=0.000001, verbose=1, epsilon=0.00001)
pretrain, test = train_test_split(indices, test_size=0.1, random_state=42, shuffle=True)
train, validation_set = train_test_split(pretrain, test_size=0.2, random_state=41)
callback = EarlyStopping(monitor='val_balanced_acc', patience=3)
model = encoding_mlp_generator(1000)

with CompressShelve("test.shelve", flag='r') as f:
    history = model.fit(generator(train,200,f),steps_per_epoch=len(train)//200, validation_data =new_generator(validation_set,200,f), validation_steps=len(validation_set)//200 , verbose = 1, epochs = 100, callbacks = [callback,rlr, mcp, tensorboard_callback], use_multiprocessing=False, workers=1)
