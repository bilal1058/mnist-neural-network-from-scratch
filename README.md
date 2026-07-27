# MNIST Neural Network From Scratch

A fully connected neural network implemented from scratch using **Python and NumPy** to classify handwritten digits from the **MNIST dataset**.

The purpose of this project is to understand how a neural network works internally rather than relying on high-level deep learning frameworks such as PyTorch or TensorFlow for the training process.

The implementation covers the complete learning pipeline, including data preprocessing, one-hot encoding, forward propagation, activation functions, softmax classification, loss gradients, backpropagation, mini-batch gradient descent, parameter updates, model evaluation, and confusion matrix analysis.

---

## Project Overview

The MNIST dataset contains grayscale images of handwritten digits from `0` to `9`.

Each image has dimensions:

```text
28 × 28 pixels
```

Each pixel contains an intensity value between:

```text
0 → black
255 → white
```

Before passing an image into the neural network, the image is flattened from:

```text
28 × 28
```

into:

```text
784
```

input values.

The network architecture used in this project is:

```text
Input Layer
784 neurons
    ↓
Hidden Layer
64 neurons
    ↓
Output Layer
10 neurons
    ↓
Predicted digit: 0-9
```

The 10 output neurons represent the 10 possible MNIST classes.

---

## Neural Network Architecture

The network contains two trainable layers.

### Layer 1

The first layer connects the 784 input pixels to 64 hidden neurons.

The weight matrix has the shape:

```text
W1 = (784, 64)
```

The bias has the shape:

```text
b1 = (1, 64)
```

The calculation is:

```text
z1 = XW1 + b1
```

The sigmoid activation is then applied:

```text
a1 = sigmoid(z1)
```

### Layer 2

The second layer connects the 64 hidden neurons to 10 output neurons.

The weight matrix has the shape:

```text
W2 = (64, 10)
```

The bias has the shape:

```text
b2 = (1, 10)
```

The calculation is:

```text
z2 = a1W2 + b2
```

Softmax is then applied to convert the output values into probabilities:

```text
a2 = softmax(z2)
```

The final prediction is the class with the highest probability.

---

# Implementation Details

## 1. Importing NumPy

```python
import numpy as np
```

NumPy is used for numerical computation and matrix operations.

The neural network relies heavily on matrix multiplication because the calculations between layers can be represented as matrix operations.

For example:

```python
np.dot(X, W1)
```

performs the matrix multiplication between the input data and the first layer's weights.

---

## 2. Loading the MNIST Dataset

```python
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

The MNIST dataset is divided into two parts:

```text
Training data:
60,000 images

Test data:
10,000 images
```

The variables are:

```text
x_train → training images
y_train → labels for training images

x_test → test images
y_test → labels for test images
```

For example:

```python
x_train[0]
```

represents the first image.

While:

```python
y_train[0]
```

represents the label of that image.

If:

```python
y_train[0] = 5
```

then the first image represents the handwritten digit `5`.

---

## 3. Flattening the Images

Initially, the training data has the shape:

```text
(60000, 28, 28)
```

This means:

```text
60000 → number of images
28    → image height
28    → image width
```

The images are flattened using:

```python
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
```

Since:

```text
28 × 28 = 784
```

each image becomes a one-dimensional vector containing 784 pixel values.

The shape changes from:

```text
(60000, 28, 28)
```

to:

```text
(60000, 784)
```

The first dimension still represents the number of images.

The second dimension now represents the 784 pixels of each image.

---

## 4. Normalizing Pixel Values

```python
x_train = x_train.reshape(60000, 784) / 255.0
x_test = x_test.reshape(10000, 784) / 255.0
```

Original MNIST pixel values range from:

```text
0 → 255
```

Dividing by `255.0` changes them to approximately:

```text
0 → 1
```

For example:

```text
0   / 255 = 0.0
128 / 255 ≈ 0.502
255 / 255 = 1.0
```

This gives the neural network smaller and more consistent input values.

---

# 5. One-Hot Encoding

The original labels are integers:

```text
0
1
2
...
9
```

For example:

```python
y_train[0] = 5
```

The network produces 10 output probabilities, one for each digit.

Therefore, the label can be represented using one-hot encoding:

```text
5
```

becomes:

```text
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
```

The index corresponding to the correct class contains `1`, while all other positions contain `0`.

The function used is:

```python
def one_hot(labels, num_classes=10):
    result = np.zeros((len(labels), num_classes))

    for i, label in enumerate(labels):
        result[i][label] = 1

    return result
```

`np.zeros()` creates an initially empty table of zeros.

For the training set:

```text
(60000, 10)
```

This means every one of the 60,000 images has a 10-element one-hot label.

The labels are converted using:

```python
y_train_oh = one_hot(y_train)
y_test_oh = one_hot(y_test)
```

---

# 6. Sigmoid Activation

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

The sigmoid function maps values into the range:

```text
0 → 1
```

It is used as the activation function for the hidden layer.

The activation introduces non-linearity into the network, allowing it to learn more complex patterns than a purely linear model.

---

# 7. Sigmoid Derivative

```python
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)
```

Backpropagation requires the derivative of the activation function.

The derivative tells us how much the output of the sigmoid changes when its input changes.

This is required when calculating how much the hidden layer contributed to the final error.

---

# 8. Softmax

```python
def softmax(x):
    exp_x = np.exp(
        x - np.max(x, axis=1, keepdims=True)
    )

    return exp_x / np.sum(
        exp_x,
        axis=1,
        keepdims=True
    )
```

Softmax converts the 10 output values into probabilities.

For example, the network might produce:

```text
[0.01, 0.02, 0.03, 0.01, 0.02,
 0.85, 0.01, 0.02, 0.01, 0.02]
```

The probabilities add up to approximately:

```text
1.0
```

The network would predict digit:

```text
5
```

because it has the highest probability.

### Why subtract the maximum?

```python
x - np.max(x, axis=1, keepdims=True)
```

This is done for numerical stability.

The exponential function can produce extremely large values. Subtracting the maximum keeps the numbers in a safer numerical range without changing the final softmax probabilities.

### Why `axis=1`?

The network processes multiple images at once.

If the output has shape:

```text
(batch_size, 10)
```

then:

```text
axis=1
```

means we operate across the 10 classes for each individual image.

### Why `keepdims=True`?

Without it, the maximum might have shape:

```text
(batch_size,)
```

With:

```python
keepdims=True
```

it remains:

```text
(batch_size, 1)
```

This allows NumPy broadcasting to correctly subtract the maximum from all 10 class values in each row.

---

# 9. Neural Network Class

```python
class NeuralNetwork:
```

The class groups everything related to the neural network into one object.

The network contains:

```text
Weights
Biases
Forward pass
Backward pass
Training process
```

---

# 10. Constructor

```python
def __init__(self, input_size, hidden_size, output_size):
```

The constructor runs automatically when a new neural network is created.

For this project:

```python
nn = NeuralNetwork(784, 64, 10)
```

Therefore:

```text
input_size  = 784
hidden_size = 64
output_size = 10
```

---

# 11. First Layer Weights

```python
self.W1 = np.random.randn(input_size, hidden_size) * 0.01
```

This creates the weights connecting the input layer to the hidden layer.

The shape is:

```text
(784, 64)
```

There are:

```text
784 × 64 = 50,176
```

weights.

The weights are initialized with small random values.

The `self` keyword stores the weights inside the network object so that other methods such as `forward()` and `backward()` can access and modify them.

---

# 12. First Layer Bias

```python
self.b1 = np.zeros((1, hidden_size))
```

This creates 64 biases.

The shape is:

```text
(1, 64)
```

The biases initially start at zero.

---

# 13. Second Layer Weights

```python
self.W2 = np.random.randn(hidden_size, output_size) * 0.01
```

This creates the weights connecting the hidden layer to the output layer.

The shape is:

```text
(64, 10)
```

There are:

```text
64 × 10 = 640
```

weights.

---

# 14. Second Layer Bias

```python
self.b2 = np.zeros((1, output_size))
```

This creates 10 output biases.

The shape is:

```text
(1, 10)
```

---

# 15. Forward Pass

```python
def forward(self, X):
```

The forward pass is the process through which an input image travels through the network to produce a prediction.

The first layer calculates:

```python
self.z1 = np.dot(X, self.W1) + self.b1
```

For a batch of 64 images:

```text
X  = (64, 784)
W1 = (784, 64)
```

Therefore:

```text
X × W1 = (64, 64)
```

The result is the input to the hidden neurons.

Next:

```python
self.a1 = sigmoid(self.z1)
```

The sigmoid activation is applied to the hidden layer.

The second layer calculates:

```python
self.z2 = np.dot(self.a1, self.W2) + self.b2
```

The shapes are:

```text
a1 = (64, 64)
W2 = (64, 10)
```

Therefore:

```text
a1 × W2 = (64, 10)
```

Finally:

```python
self.a2 = softmax(self.z2)
```

converts the 10 output values into probabilities.

The prediction is returned:

```python
return self.a2
```

---

# 16. Backward Pass

```python
def backward(self, X, y_true, learning_rate=0.1):
```

The backward pass calculates gradients and updates the weights.

The goal is to determine:

```text
How should each weight change
to reduce the prediction error?
```

First:

```python
m = X.shape[0]
```

gets the number of examples in the current mini-batch.

If the batch contains 64 images:

```text
m = 64
```

---

## Output Error

```python
dz2 = self.a2 - y_true
```

This calculates the output error.

For the correct class, the network wants the probability to increase.

For incorrect classes, it wants the probabilities to decrease.

The result has shape:

```text
(batch_size, 10)
```

---

## Gradient of W2

```python
dW2 = np.dot(self.a1.T, dz2) / m
```

This calculates how much each weight in `W2` contributed to the error.

The gradients are averaged over the mini-batch by dividing by `m`.

The resulting shape is:

```text
(64, 10)
```

which matches:

```text
W2 = (64, 10)
```

---

## Gradient of b2

```python
db2 = np.sum(dz2, axis=0, keepdims=True) / m
```

This calculates the gradient for the output biases.

The errors from all examples in the batch are summed for each output neuron and then averaged.

The resulting shape is:

```text
(1, 10)
```

---

## Hidden Layer Error

```python
dz1 = np.dot(dz2, self.W2.T) * sigmoid_derivative(self.z1)
```

This propagates the error backward from the output layer to the hidden layer.

The network uses the chain rule to determine how much the hidden layer contributed to the final error.

The sigmoid derivative is included because the hidden layer uses sigmoid activation.

---

## Gradient of W1

```python
dW1 = np.dot(X.T, dz1) / m
```

This calculates how much each weight in `W1` contributed to the error.

The resulting shape is:

```text
(784, 64)
```

which matches:

```text
W1 = (784, 64)
```

---

## Gradient of b1

```python
db1 = np.sum(dz1, axis=0, keepdims=True) / m
```

This calculates the gradient of the hidden-layer biases.

The shape is:

```text
(1, 64)
```

---

# 17. Updating the Parameters

The weights are updated using gradient descent.

```python
self.W1 -= learning_rate * dW1
self.b1 -= learning_rate * db1
self.W2 -= learning_rate * dW2
self.b2 -= learning_rate * db2
```

The basic idea is:

```text
New parameter
=
Old parameter
-
Learning rate × Gradient
```

The subtraction moves the parameters in the direction that reduces the error.

The learning rate controls how large each update is.

---

# 18. Mini-Batch Training

```python
def train(self, X, y, epochs=10, batch_size=64, learning_rate=0.1):
```

This method trains the network on the complete dataset.

The parameters mean:

```text
epochs = 10
batch_size = 64
learning_rate = 0.1
```

---

## Number of Training Samples

```python
num_samples = X.shape[0]
```

For MNIST:

```text
num_samples = 60,000
```

---

## Epoch Loop

```python
for epoch in range(epochs):
```

One epoch means the model processes the entire training dataset once.

With:

```text
epochs = 10
```

the model sees the training dataset 10 times.

---

## Shuffle the Data

```python
indices = np.random.permutation(num_samples)
```

This creates randomly shuffled indices.

The data is then shuffled:

```python
X_shuffled = X[indices]
y_shuffled = y[indices]
```

Shuffling prevents the model from always seeing the data in the same order.

---

## Create Mini-Batches

```python
for i in range(0, num_samples, batch_size):
```

With 60,000 images and a batch size of 64, the model processes the dataset in mini-batches.

Each batch contains up to 64 images.

```python
X_batch = X_shuffled[i:i+batch_size]
y_batch = y_shuffled[i:i+batch_size]
```

The corresponding images and labels are selected together.

---

## Forward and Backward Pass

```python
self.forward(X_batch)
self.backward(X_batch, y_batch, learning_rate)
```

For each mini-batch:

```text
64 images
    ↓
Forward pass
    ↓
Predictions
    ↓
Calculate gradients
    ↓
Average gradients
    ↓
Update weights
```

This process repeats until all 60,000 training images have been processed.

---

# 19. Calculate Training Accuracy

After each epoch:

```python
predictions = self.forward(X)
```

The network makes predictions for the entire training set.

Then:

```python
np.argmax(predictions, axis=1)
```

selects the predicted digit with the highest probability.

The actual labels are also obtained using:

```python
np.argmax(y, axis=1)
```

Then:

```python
accuracy = np.mean(
    np.argmax(predictions, axis=1) ==
    np.argmax(y, axis=1)
)
```

calculates the percentage of correct predictions.

---

# 20. Testing the Model

After training, the model is evaluated on unseen test data.

```python
test_preds = nn.forward(x_test)
```

The network produces predictions for the 10,000 test images.

Then:

```python
test_predictions = np.argmax(test_preds, axis=1)
```

converts the 10 output probabilities for each image into a single predicted digit.

Finally:

```python
test_accuracy = np.mean(test_predictions == y_test)
```

calculates the test accuracy.

The test set is important because the model never trained on these images.

This gives a better indication of how well the network generalizes to unseen data.

---

# 21. Confusion Matrix

A confusion matrix shows how the model classifies each digit.

The rows represent:

```text
Actual labels
```

The columns represent:

```text
Predicted labels
```

The diagonal represents correct predictions.

For example:

```text
Actual 5 → Predicted 5
```

is a correct classification.

An off-diagonal value such as:

```text
Actual 5 → Predicted 3
```

represents a mistake.

The confusion matrix helps identify which digits the model finds difficult to distinguish.

---

# 22. Visualizing Predictions

The project also visualizes predictions on individual test images.

For each image, the visualization displays:

```text
True label
Predicted label
Confidence
```

The predicted class is obtained using:

```python
predicted = np.argmax(pred)
```

The confidence is calculated using:

```python
confidence = np.max(pred) * 100
```

The image is then displayed using Matplotlib.

Correct predictions can be visually distinguished from incorrect predictions.

---

# 23. Important Concepts Learned

This project demonstrates several fundamental neural network concepts:

### Parameters

The trainable values of the network:

```text
W1
b1
W2
b2
```

### Forward Propagation

Passing input data through the network to generate predictions.

### Activation Function

Sigmoid introduces non-linearity into the hidden layer.

### Softmax

Converts output values into probabilities across the 10 digit classes.

### Backpropagation

Propagates the prediction error backward through the network to calculate gradients.

### Gradient

The gradient indicates how changing a parameter affects the error.

### Gradient Descent

Uses the gradients to update parameters in a direction that reduces error.

### Mini-Batch Gradient Descent

Instead of processing the entire dataset at once, the model processes small batches and averages the gradients within each batch before updating the parameters.

### Epoch

One complete pass through the training dataset.

### Generalization

The ability of the trained model to correctly classify previously unseen test images.

---

# Results

The notebook reports:

* Training accuracy after each epoch
* Test accuracy on unseen MNIST images
* Confusion matrix
* Correct and incorrect predictions
* Prediction confidence

Add your actual final result here after running the notebook:

```text
Training Accuracy: XX.XX%
Test Accuracy: XX.XX%
Epochs: 10
Batch Size: 64
Hidden Neurons: 64
Learning Rate: 0.6
```

---

# Technologies Used

* Python
* NumPy
* Matplotlib
* TensorFlow Keras MNIST Dataset
* Scikit-learn
