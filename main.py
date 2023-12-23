import random
import pickle
import numpy as np
from deep_learn import Dense, Model
import matplotlib.pyplot as plt

VALIDATION_SPLIT = 0.8


def results(model, false_data_idx: list, data: np.ndarray):
    """Display both correct and false predictions. False prediction indices are required. """

    np.random.shuffle(false_data_idx)
    false_data_idx = false_data_idx[:9]
    correct_idx = []
    while 1:
        i = random.randint(0, data.shape[0])
        if i not in correct_idx and i not in false_data_idx:
            correct_idx.append(i)
        if len(correct_idx) == 9:
            break

    correct_labels = np.argmax(model.predict(data[correct_idx, :].T), axis=0)
    false_labels = np.argmax(model.predict(data[false_data_idx, :].T), axis=0)


    data = data.reshape((-1, 28, 28)) * 255
    titles = ['Correct Images', 'False Images']

    for i, dataset in enumerate([data[correct_idx, :, :], data[false_data_idx, :, :]]):
        fig, ax = plt.subplots(3, 3, figsize=(7, 7))
        fig.suptitle(titles[i])
        for y in range(3):
            for x in range(3):
                ax[y][x].imshow(dataset[(y * 3)+x, :, :])
                ax[y][x].set_title(f'Correct label: {correct_labels[(y * 3)+x]}' if i == 0 else f"False Label: {false_labels[(y * 3)+x]}")
        plt.tight_layout()
        plt.show()


"""Prepare dataset for learning process"""
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train, y_train = x_train.reshape((-1, 784)) / 255., keras.utils.to_categorical(y_train, num_classes=10)
x_validation = x_test[int(10000*(1-VALIDATION_SPLIT)):].reshape((-1, 784)) / 255.
y_validation = keras.utils.to_categorical(y_test[int(10000*(1-VALIDATION_SPLIT)):], num_classes=10)
x_test = x_test[:int(10000*(1-VALIDATION_SPLIT))].reshape((-1, 784)) / 255.
y_test = keras.utils.to_categorical(y_test[:int(10000*(1-VALIDATION_SPLIT))], num_classes=10)


def train_model():
    """ Train new model """
    seed = 11
    layers = [Dense(neurons_num=512, activation='relu', input_shape=784, name='layer_1', seed=seed),
              Dense(neurons_num=128, activation='relu', name='layer_2', seed=seed),
              Dense(neurons_num=10, activation='sigmoid', name='output_layer', seed=seed)]

    learning_rates = [0.5, 0.3, 0.2]

    for layer_id in range(len(layers)):
        layers[layer_id].compile(learning_rates[layer_id], initialization='he')

    model = Model(layers)
    model.compile(loss_function='cross_entropy', optimization='momentum', momentum_beta=0.9)
    model.fit(x=x_train, y=y_train, epochs=30, batch_size=64,
              validation_data=x_validation, validation_label=y_validation)

    model.display(model.test_predict(test_data=x_test, test_label=y_test))

    model.save_model('model.plk')
    return model


"""Train and save new model"""
model = train_model()

"""Load already existing model"""
model = pickle.load(open('model.plk', 'rb'))

"""Evaluate model on test data"""
pred = model.predict(x_test.T)




false_idx_images = []
for idx in range(pred.shape[1]):
    if np.argmax(pred.T[idx]) != np.argmax(y_test[idx]):
        false_idx_images.append(idx)

print('Test error: ', np.round(len(false_idx_images) / size[1], 3))

results(model, false_data_idx=false_idx_images, data=x_test)
