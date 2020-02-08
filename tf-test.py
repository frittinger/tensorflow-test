from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

def print_example_image(img):
# print example image
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def print_example_data(train_images, train_labels, class_names):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()


def info(train_images, train_labels, test_images, test_labels) :
    print("# tensorflow version: " + tf.__version__)
    print("# train_images.shape: {0}".format(train_images.shape))
    print("# number of training labels: {0}".format(len(train_labels)))


def plot_image(i, predictions_array, true_label, img, class_names):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
                                         color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def verify_predictions(i, predictions, test_labels, test_images, class_names):
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(i, predictions[i], test_labels, test_images, class_names)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions[i],  test_labels)
    plt.show()


def print_verify_predictions(s, n, predictions, test_labels, class_names):

    for i in range(s, n):
        predicted_label = np.argmax(predictions[i])
        actual_label = test_labels[i]
        if predicted_label == actual_label:
            print("## {0} correct: {1}".format(i, class_names[actual_label]))
        else:
            print("## {0} wrong: {1} != {2}".format(i, class_names[predicted_label], class_names[actual_label]))


def main() :
    
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    info(train_images, train_labels, test_images, test_labels)

    #print_example_image(train_images[0])

    # prepare data
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    #print_example_data(train_images, train_labels, class_names)

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)
        ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=1)

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)

    predictions = model.predict(test_images)

    print_verify_predictions(100, 120, predictions, test_labels, class_names)

    # Plot the first X test images, their predicted labels, and the true labels.
    # Color correct predictions in blue and incorrect predictions in red.
    # num_rows = 5
    # num_cols = 3
    # num_images = num_rows*num_cols
    # plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    # for i in range(num_images):
    #     j = i * 2
    #     plt.subplot(num_rows, 2*num_cols, 2*i+1)
    #     plot_image(j, predictions[j], test_labels, test_images, class_names)
    #     plt.subplot(num_rows, 2*num_cols, 2*i+2)
    #     plot_value_array(j, predictions[j], test_labels)
    # plt.tight_layout()
    # plt.show()

    # verify_predictions(1, predictions, test_labels, test_images, class_names)
    # verify_predictions(10, predictions, test_labels, test_images, class_names)
    # verify_predictions(100, predictions, test_labels, test_images, class_names)
    # verify_predictions(1000, predictions, test_labels, test_images, class_names)


main()

