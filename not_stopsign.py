import cv2
import glob
import numpy as np
# import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

# importing training data
stop_sign = glob.glob('OwnCollection/stopsigns/*.jpg')
no_sign = glob.glob('OwnCollection/non-vehicles/**/*.png')

# resizing the pics :)
# j = 0
# for i in stop_sign:
#     image = cv2.imread(i)
#     try:
#         image = cv2.resize(np.float32(image), (64, 64))
#         cv2.imwrite('OwnCollection/stopsigns64/{}.jpg'.format(j), image)
#         j += 1
#     except:
#         j -= 1

stop_sign64 = glob.glob('OwnCollection/stopsigns64/*.jpg')
# # print(stop_sign64[20])
# image_test = plt.imread(stop_sign64[20])
# plt.imshow(image_test)
# plt.waitforbuttonpress(0)

data = []
stop_features = []
for i in stop_sign64:
    image = cv2.imread(i)
    stop_features.append(image)
    data.append(image)

nonstop_features = []
for i in no_sign:
    image = cv2.imread(i)
    nonstop_features.append(image)
    data.append(image)

# print(stop_features)
stop_features = np.array(stop_features)
stop_labels = np.ones(len(stop_features))
# print(stop_labels.shape)
# print(stop_features.shape)

nonstop_features = np.array(nonstop_features)
nonstop_labels = np.zeros(len(nonstop_features))
# print(nonstop_features.shape)

data = np.array(data)
labels = np.concatenate((stop_labels, nonstop_labels))
# print(data.shape)
# print(data)
# print(labels)

data = data/255.0
train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.25, random_state=42)

# creating model
image_shape = (64, 64, 3)
# height, width, depth = data.shape
# image_shape = (height, width, depth)
# if K.image_data_format() == "channels_first":
#     image_shape = (depth, height, width)

cnn_model = Sequential()
cnn_model.add(Conv2D(20, (3, 3), input_shape=image_shape, activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Conv2D(50, (3, 3), input_shape=image_shape, activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(output_dim=32, activation='relu'))
cnn_model.add(Dense(output_dim=2, activation='sigmoid'))

cnn_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

history = cnn_model.fit(train_x,
                        train_y,
                        batch_size=500,
                        nb_epoch=50,
                        verbose=1,
                        validation_data=(test_x, test_y))

score = cnn_model.evaluate(test_x, test_y, verbose=0)
print('Test Accuracy : {:.4f}'.format(score[1]))
