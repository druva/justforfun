import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import math
import numpy as np
import matplotlib.pyplot as plt

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

def initDataSet(verbose):
    #load data set and test set
    dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']
    class_names = metadata.features['label'].names
    num_train_examples = metadata.splits['train'].num_examples
    num_test_examples = metadata.splits['test'].num_examples
    if verbose:
        print("Class names: {}".format(class_names))
        print("Number of training examples: {}".format(num_train_examples))
        print("Number of test examples:     {}".format(num_test_examples))
    train_dataset, test_dataset = doNormalize(train_dataset, test_dataset, verbose)
    return dataset, metadata, train_dataset, test_dataset, class_names, num_train_examples, num_test_examples

def normalize(images, labels):
  # print("A", images)
  images = tf.cast(images, tf.float32)
  # print("B", images)
  images /= 255
  # print("C", images)
  return images, labels

def doNormalize(train_dataset, test_dataset, verbose):
    if verbose:
        print("train_dataset-Before", train_dataset)
        print("test_dataset-Before", test_dataset)

    train_dataset =  train_dataset.map(normalize)
    test_dataset  =  test_dataset.map(normalize)

    if verbose:
        print("train_dataset-After", train_dataset)
        print("test_dataset-After", test_dataset)
    return train_dataset, test_dataset

def viewOneImage(dataset, verbose):
    if verbose:
        for image, label in dataset.take(1):
            break;
        image = image.numpy().reshape((28,28))

        plt.figure()
        plt.imshow(image, cmap=plt.cm.binary)
        plt.colorbar()
        plt.grid(False)
        plt.show()

def viewMoreImages(dataset, class_names, verbose):
    if verbose:
        plt.figure(figsize=(10,10))
        for i, (image, label) in enumerate(dataset.take(25)):
            image = image.numpy().reshape((28,28))
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(image, cmap=plt.cm.binary)
            plt.xlabel(class_names[label])
        plt.show()

def createModelBase():
    print("createModelBase")
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    return model

def createModelCNN():
    print("createModelCNN")
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu,
                               input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    return model

def compileModel(model):
    print("compileModel")
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
    return model

def showLoss(history):
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss Magnitude')
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.show()

def plot_image(i, predictions_array, true_labels, images, class_names):
  predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img[...,0], cmap=plt.cm.binary)

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
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

def run1(itr):
    verbose = True
    dataset, metadata, train_dataset, test_dataset, class_names, num_train_examples, num_test_examples = initDataSet(verbose)
    # viewOneImage(test_dataset, verbose)
    # viewMoreImages(test_dataset, class_names, verbose)
    model = createModelCNN()
    model = compileModel(model)


    BATCH_SIZE = 64
    print(train_dataset)
    train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
    print(train_dataset)
    test_dataset = test_dataset.cache().batch(BATCH_SIZE)

    #What is epochs? why 5 value
    #what is fit?
    steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE)
    for i in range(itr):
        print(f"Running Fit ({i+1})n times, steps_per_epoch={steps_per_epoch}")
        history = model.fit(train_dataset, epochs=8, steps_per_epoch=steps_per_epoch)
    showLoss(history)
    return BATCH_SIZE, test_dataset, class_names, num_test_examples, model

def run2(itr):
    print("run2", itr)
    verbose = True
    dataset, metadata, train_dataset, test_dataset, class_names, num_train_examples, num_test_examples = initDataSet(verbose)
    # viewOneImage(test_dataset, verbose)
    # viewMoreImages(test_dataset, class_names, verbose)
    model = createModelBase()
    model = compileModel(model)


    BATCH_SIZE = 64
    train_dataset = train_dataset.cache() \
                                .batch(BATCH_SIZE)  # No shuffle, no repeat

    test_dataset = test_dataset.cache().batch(BATCH_SIZE)

    for i in range(itr):
        print(f"Running Fit ({i+1})n times")
        history = model.fit(train_dataset, epochs=10)
    showLoss(history)
    return BATCH_SIZE, test_dataset, class_names, num_test_examples, model

def run():
    print("run1")
    #Experiment 1
    BATCH_SIZE, test_dataset, class_names, num_test_examples, model = run1(1)
    #Experiment 2
    # model = run2()

    test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/BATCH_SIZE))
    print('Accuracy on test dataset:', test_accuracy)


    for test_images, test_labels in test_dataset.take(1):
        test_images = test_images.numpy()
        test_labels = test_labels.numpy()
        predictions = model.predict(test_images)

    # Plot the first X test images, their predicted label, and the true label
    # Color correct predictions in blue, incorrect predictions in red
    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions, test_labels, test_images, class_names)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions, test_labels)

run1(2)
