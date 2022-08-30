import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

#print(tf.__version__)

num_rows = 5
num_cols = 3
num_images = num_cols*num_rows

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

#print(np.argmax(predictions[0]))

#print(test_label[0])

def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap = plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
                                         color = color
                                         )
def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color = '#777777')
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

#plt.figure(figsize=(2*2*num_cols, 2*num_rows))
#for i in range(num_images):
#    plt.subplot(num_rows, 2*num_cols, 2*i+1)
#    plot_image(i, predictions[i], test_labels, test_images)
#    plt.subplot(num_rows, 2*num_cols, 2*i+2)
 #   plot_value_array(i, predictions[i], test_labels)
#plt.tight_layout()
#plt.show()
img = test_images[1]
img = (np.expand_dims(img,0))
#print(img.shape)

prediction_single = probability_model.predict(img)

#print(prediction_single)
plot_image(1, prediction_single[0], test_labels, test_images)
plt.show()
plot_value_array(1, prediction_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation = 45)
plt.show()