from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
import numpy as np
import argparse
import cv2
import os
from imutils import paths

class Architecture:
    @staticmethod
    def build(width, height, depth, classes, weightsPath):
        model = Sequential()
        
        model.add(Convolution2D(max(classes * 2, 20), 5, 5, border_mode = "same", input_shape = (depth, height, width), activation = "relu"))
        model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering = "th"))
        
        model.add(Convolution2D(max(classes * 4, 50), 5, 5, border_mode = "same", activation = "relu"))
        model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering = "th"))
        
        model.add(Convolution2D(max(classes * 8, 50), 5, 5, border_mode = "same", activation = "relu"))
        #model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering = "th"))
        
        model.add(Convolution2D(max(classes * 16, 50), 5, 5, border_mode = "same", activation = "relu"))
        #model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering = "th"))
        
        model.add(Flatten())
        #model.add(Dense(classes * 16))
        #model.add(Activation("relu"))
        model.add(Dense(classes * 4))
        model.add(Activation("relu"))
        
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        if weightsPath is not None:
            model.load_weights(weightsPath)
        return model

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--save-model", type = int, default = -1, help="(optional) whether or not model should be saved to disk")
parser.add_argument("-l", "--load-model", type = int, default = -1, help="(optional) whether or not pre-trained model should be loaded")
parser.add_argument("-w", "--weights", type = str, help = "(optional) path to store weights")
parser.add_argument("-d", "--dataset", type = str, required = True, help = "path to the dataset")
args = vars(parser.parse_args())

print("[STATUS] Dataset loading...\n\n")
imagePaths = list(paths.list_images(args["dataset"]))

data = [[[[]]]]
labels = []
edgeDetect = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
for(i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]
    filtered = cv2.filter2D(src = image, ddepth = -1, kernel = edgeDetect)
    features = cv2.resize(filtered, (32, 32))
    data.append(features)
    labels.append(label)
    
    if i > 0 and i % 1000 == 0:
        print("PROCESSED {}/{}".format(i, len(imagePaths)))

LE = LabelEncoder()
labels = LE.fit_transform(labels)
labels = np_utils.to_categorical(labels, 10)
data = float(np.array(data)) / 255.0
(trainData, testData, trainLabels, testLabels) = train_test_split(data, labels, test_size = 0.20)

trainLabels = np_utils.to_categorical(trainLabels, 10)
testLabels = np_utils.to_categorical(testLabels, 10)

print("[STATUS] Model is being compiled...")
model = Architecture.build(width = 32, height = 32, depth = 3, classes = 2, weightsPath = args["weights"] if args["load_model"] > 0 else None)
model.compile(loss = "binary_crossentropy", optimizer = SGD(lr = 0.015), metrics = ["accuracy"])

if args["load_model"] < 0:
    print("TRAINING WEIGHTS...")
    model.fit(trainData, trainLabels, batch_size=128, epochs=5, verbose=1)
    print("EVALUATING DATA...")
    (loss, accuracy) = model.evaluate(testData, testLabels, batch_size=128, verbose=1)
    print("ACCURACY >> {:.4f}%".format(accuracy * 100))

if args["save_model"] > 0:
    print("Weights being stored for future use...")
    model.save_weights(args["weights"], overwrite=True)

probabilities = model.predict(testData[np.newaxis, i])
prediction = probabilities.argmax(axis = 1)
    
print("PREDICTED: {}, ACTUAL: {}".format(prediction[0], np.argmax(testLabels[i])))

