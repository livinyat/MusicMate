import json
from tabnanny import verbose
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras as keras 

DATASET_PATH = "data_original.json"

def load_data(dataset_path) : 
    with open(dataset_path , "r") as fp : 
        data = json.load(fp)

    #convert lists into numpy arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs , targets

def prepare_datasets(test_size , validation_size) :
    # load data
    inputs , targets = load_data(DATASET_PATH)

    # create train/test split 
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs , targets , test_size=test_size)

    # create train/validiation split
    inputs_train, inputs_validation, targets_train, targets_validation = train_test_split(inputs_train, targets_train, test_size = validation_size)

    # 3d array
    inputs_train = inputs_train[... , np.newaxis]
    inputs_validation = inputs_validation[... , np.newaxis]
    inputs_test = inputs_test[... , np.newaxis]

    return inputs_train , inputs_validation , inputs_test , targets_train , targets_validation , targets_test

def build_model(input_shape) :
    # create model
    model = keras.Sequential()

    # 1st convolutional layer
    model.add(keras.layers.Conv2D(32, (3,3) , activation="relu" , input_shape = input_shape))
    model.add(keras.layers.MaxPool2D((3,3) , strides=(2,2) , padding = "same"))
    model.add(keras.layers.BatchNormalization())

    # 2nd convolutional layer
    model.add(keras.layers.Conv2D(32, (3,3) , activation="relu" , input_shape = input_shape))
    model.add(keras.layers.MaxPool2D((3,3) , strides=(2,2) , padding = "same"))
    model.add(keras.layers.BatchNormalization())

    # 3rd convolutional layer
    model.add(keras.layers.Conv2D(32, (2,2) , activation="relu" , input_shape = input_shape))
    model.add(keras.layers.MaxPool2D((2,2) , strides=(2,2) , padding = "same"))
    model.add(keras.layers.BatchNormalization())

    # flatten the output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation = "relu"))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10 , activation = "softmax"))

    return model 

def predict(model , inputs , targets) :
    inputs = inputs[np.newaxis, ...]

    # prediction = [[] , []]
    prediction = model.predict(inputs)

    # extract index with max value
    predicted_index = np.argmax(prediction , axis = 1)
    print("Expected index : {}, predicted index: {}".format(targets , predicted_index))

if __name__ == "__main__" : 
    # create train , validate and test sets
    inputs_train , inputs_validate , inputs_test , targets_train , targets_validate , targets_test = prepare_datasets(0.25, 0.2)

    # build the CNN network
    input_shape = (inputs_train.shape[1] , inputs_train.shape[2] , inputs_train.shape[3])
    model = build_model(input_shape)

    # compile the CNN network
    optimizer = keras.optimizers.Adam(learning_rate = 0.0001)
    model.compile(optimizer=optimizer, loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

    # train the CNN network
    model.fit(inputs_train, targets_train, validation_data=(inputs_validate,targets_validate), batch_size = 32 , epochs=50)

    # evaluate the CNN on test set
    test_error , test_accuracy = model.evaluate(inputs_test , targets_test , verbose = 1)
    print("Accuracy on the test set is: {}".format(test_accuracy))

    # make prediction on a sample
    inputs = inputs_test[100]
    targets = targets_test[100]

    predict(model , inputs , targets)