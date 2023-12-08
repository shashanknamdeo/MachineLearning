"""
Types of Neural Network in Machine Learning:

1. ANN : ANN is also known as an artificial neural network. It is a feed-forward neural network because the inputs are sent in the forward direction. It can also contain hidden layers which can make the model even denser. They have a fixed length as specified by the programmer. It is used for Textual Data or Tabular Data. A widely used real-life application is Facial Recognition. It is comparatively less powerful than CNN and RNN.

2. CNN : Convolutional Neural Networks is mainly used for Image Data. It is used for Computer Vision. Some of the real-life applications are object detection in autonomous vehicles. It contains a combination of convolutional layers and neurons. It is more powerful than both ANN and RNN.

3. RNN : It is also known as Recurrent Neural Networks. It is used to process and interpret time series data. In this type of model, the output from a processing node is fed back into nodes in the same or previous layers. The most known types of RNN are LSTM (Long Short Term Memory) Networks

->  floating point values are more suitable for modeling with a neural network.

Smaple : A sample is a single row of data.It contains inputs that are fed into the algorithm and an output that is used to compare to the prediction and calculate an error. A training dataset is comprised of many rows of data, e.g. many samples.A sample may also be called an instance, an observation, an input vector, or a feature vector.

Batch size : The batch size is a hyperparameter that defines the number of samples to work through before updating the internal model parameters. Think of a batch as a for-loop iterating over one or more samples and making predictions. At the end of the batch, the predictions are compared to the expected output variables and an error is calculated. From this error, the update algorithm is used to improve the model, e.g. move down along the error gradient.

Epochs : The number of epochs is a hyperparameter that defines the number times that the learning algorithm will work through the entire training dataset. One epoch means that each sample in the training dataset has had an opportunity to update the internal model parameters.

Optimizers : In deep learning, optimizers are algorithms that adjust the model’s parameters during training to minimize a loss function. They enable neural networks to learn from data by iteratively updating weights and biases.

Dropout : The term “dropout” refers to dropping out the nodes (input and hidden layer) in a neural network (as seen in Figure 1). All the forward and backwards connections with a dropped node are temporarily removed, thus creating a new network architecture out of the parent network. The nodes are dropped by a dropout probability of p. Dropout = 0.5 is the most optimised keep probability, that states dropping 50% of the nodes.


"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

