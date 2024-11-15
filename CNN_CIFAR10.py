import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import ssl

# turn off ssl verification so we can download the data
ssl._create_default_https_context = ssl._create_unverified_context

# Dataset of 50,000 32x32 color training images and 10,000 test images
# Labeled: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
# See some of the images here: https://www.cs.toronto.edu/~kriz/cifar.html
# More about it here: https://keras.io/api/datasets/cifar10/
from keras.datasets import cifar10

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
Final_X_test = X_test[-1]
Final_Y_test = Y_test[-1]
X_test = X_test[:-1]
Y_test = Y_test[:-1]

# Transform pixel values to be between 0 and 1
X_train, X_test = X_train/255.0, X_test/255.0

# The first layer will have 50 filters. The filter size is 2 X 2
# Max pooling is then used to reduce the spatial dimensions of 
#  the output volume on the next line.
CNN_model = models.Sequential()
CNN_model.add(layers.Conv2D(50, (2, 2), activation='relu', input_shape=(32, 32, 3)))
CNN_model.add(layers.MaxPooling2D((3, 3)))
CNN_model.add(layers.Flatten())
CNN_model.add(layers.Dense(100, activation='relu')) # changed 50 to 100
CNN_model.add(layers.Dropout(.1))
CNN_model.add(layers.Dense(10, activation='softmax'))


# By increasing the dense layer size it increased the capacity of the model to capture more complex patterns 
# and allows the model to learn more

# Reducing the learning rate seems to have helped the model converge more effectively during training. 
# A smaller learning rate allows the optimizer to take smaller steps towards the optimal solution, 
# which can help the model to escape local minima and converge to a better optimum. 
# It also helps stabilize the training process and prevent overfitting.

# By making these changes, I provided the model with more capacity to learn complex patterns

optimizer = tf.optimizers.Adam(learning_rate = .001) # changed 0.005 to 0.001

CNN_model.compile(optimizer=optimizer,
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), # used when each input belongs to one class; The network will output a set of probabilities for each class for ech image
          metrics=['accuracy'])

history = CNN_model.fit(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test))


#Evaluate the model
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='upper left')
test_loss, test_acc = CNN_model.evaluate(X_test,  Y_test, verbose=2)
print(test_acc)


#show model’s structure
CNN_model.summary()

#create an array that predicts what the category of each image
#from the test_image array and then print out the prediction for
#the 10th image.
predictions = CNN_model.predict(np.array([Final_X_test]))
tenth_prediction = (predictions[0])
print(tenth_prediction)

#The output from the code above consists of probability
#distributions between zero and one. 

import numpy as np
#print out the index of the array which contains the highest probability distribution
prediction = (np.argmax(predictions[0]))
print(prediction)


#The index with the highest number will be the predicted output. 
#The indexes of our input data are as follows:

#  0	airplane
#  1	automobile
#  2	bird
#  3	cat
#  4	deer
#  5	dog
#  6	frog
#  7	horse
#  8	ship
#  9	truck

#so if the returned index is 0, then the CNN categorized the inputted image as an “airplane”

plt.figure()
plt.imshow(Final_X_test)
plt.colorbar()
plt.grid(False)
plt.show()
