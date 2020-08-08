import tensorflow as tf
import numpy as np
mnist = tf.keras.datasets.mnist

# Loading the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizing the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape and format
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Neural network configuration
dense_layers = [1]
conv_layers = [2]
perceptron_units = [256]

# Training the model
for dense_layer in dense_layers:
    for conv_layer in conv_layers:
        for perceptron_unit in perceptron_units:

            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Conv2D(
                32, (3, 3), input_shape=(28, 28, 1)))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))

            for i in range(conv_layer - 1):
                model.add(tf.keras.layers.Conv2D(32, (3, 3)))
                model.add(tf.keras.layers.MaxPooling2D((2, 2)))

            model.add(tf.keras.layers.Flatten())

            for i in range(dense_layer):
                model.add(tf.keras.layers.Dense(
                    perceptron_unit, activation='relu'))

            model.add(tf.keras.layers.Dropout(0.2))

            model.add(tf.keras.layers.Dense(10, activation='softmax'))

            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            model.fit(
                x_train,
                y_train,
                epochs=5)

            model.evaluate(x_test, y_test)

            model.save('model1')
