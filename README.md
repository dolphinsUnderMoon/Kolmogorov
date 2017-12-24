# Kolmogorov

Some simple implementations using PyTorch. 

The project's name *Kolmogorov* pays tribute to the Russian mathematician Andrey Nikolaevich Kolmogorov.

## CNN

`cnn.py` implemented a lightweight CNN which has two convolutional layers to classificate the mnist images. The `cnn_last_layer_visual.png` in `Visualizing` directory visualizes the last layer of the CNN.

## RNN

There are two RNNs, one for MNIST classification and another for regression.

The `rnn_mnist.py` get mnist image as input, which each row is a time step. LSTM is used in this NN.

The `rnn_regression.py` get `sin(x)` as input and predicts corressponding `cos(x)` value. 

## GAN

`gan.py` implemented a GAN which generator tries to generates a quadratic curve.

## AutoEncoder

`autoencoder.py` encoded the mnist images in 3-dim vector and `ae_embeddings_visual.png` in `Visualizing` directory shows each number's embeddings in a 3-D space. If you do not have the MNIST data set, just modifies the`is_mnist_exist` in  `line 17` of the source code to download it.