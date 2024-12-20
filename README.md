# Neural Network From Scratch #
This repository demonstrates the implementation of a neural network entirely from scratch using Python. It covers foundational concepts, including custom activation functions, layers, loss functions, and training on datasets like XOR and MNIST.

# Features 
Custom Layers: Fully implemented dense and convolutional layers.
Activation Functions: Includes Tanh, ReLU, and more.
Loss Functions: Mean Squared Error (MSE) and others.
Dataset Examples: Train the network on XOR and MNIST datasets.
Modular Design: Easily extendable architecture.
Quick Start
Run XOR Example
```
python3 xor.py
```
Train on MNIST
```
python3 mnist.py
```
# XOR dataset
X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

# Model definition
layers = [Dense(2, 3), Tanh(), Dense(3, 1), Tanh()]
train(layers, mse, mse_prime, X, Y, epochs=1000, learning_rate=0.1)
# File Structure
activation.py: Defines individual activation functions.
dense.py: Implements fully connected (dense) layers.
convolutional.py: Implements convolutional layers.
losses.py: Contains loss functions like MSE.
network.py: Core network logic, including training methods.
mnist.py: Script for training on the MNIST dataset.
xor.py: Script for training on the XOR dataset.
