import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

#print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist             #using fashion MNIST dataset which has 70k images
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#assigning class names for reference and labelling
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

print("Training dataset data:")
#Checking data of the training data:
#in this case there are 60k images of 28x28 resolution
print(train_images.shape)
#hence there are 60k labels in the dataset with values 0 to 9
print(len(train_labels))
print(train_labels)

print("Test dataset data:")
#Checking data of the test set:
#in this case there are 10k images of 28x28 resolution
print(test_images.shape)
#hence there are 10k labels in the dataset with values 0 to 9
print(len(train_labels))
print(train_labels)

#setting the colour values which range from 0 to 255 to a range 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

#inspecting the first 25 images in the set
plt.figure(figsize=(10,10))
#for i in range(25):
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(train_images[i], cmap = plt.cm.binary)
#    plt.xlabel(class_names[train_labels[i]])
#plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28,28)),
    #reformats data from 2 dimensional to 1 dimensional array
    #flatten basically takes the individual rows and lines them up
    tf.keras.layers.Dense(128, activation = 'relu'),
    #actual layer which has parameters to learn
    #basically the number means the number of neurons it'll have
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics = ['accuracy'])

model.fit(train_images, train_labels, epochs = 10)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print("\n Test accuracy:", test_acc)

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

print(np.argmax(predictions[0]))