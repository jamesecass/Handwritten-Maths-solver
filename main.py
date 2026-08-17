import numpy as np
from keras.datasets import mnist  

LEARNING_RATE = 0.1

(input_train, target_train), (input_test, target_test) = mnist.load_data()

input_train = input_train.reshape(input_train.shape[0], 784) / 255.0
input_test = input_test.reshape(input_test.shape[0], 784) / 255.0

class NeuralNetwork:
    def __init__(self,input_size=784,hidden_size=10,output_size=10):
        self.Weights_hidden = np.random.randn(hidden_size,input_size) * 0.01
        self.Bias_hidden = np.zeros((hidden_size,1))

        self.Weights_output = np.random.randn(output_size,hidden_size) * 0.01
        self.Bias_output = np.zeros((output_size,1))

    def forward_prop(self,Input):
        Weighted_Sum1 = self.Weights_hidden.dot(Input) + self.Bias_hidden
        Anwser_1 = ReLu(Weighted_Sum1)
        Weighted_Sum2 = self.Weights_output.dot(Anwser_1) + self.Bias_output

        return Weighted_Sum1, Anwser_1, Weighted_Sum2

    def back_prop(self, anwser_index, Weighted_Sum1, Anwser_1, Weighted_Sum2, Input):
        one_hot_vector = one_hot(anwser_index)
        output_error = Weighted_Sum2 - one_hot_vector

    def train(self):
        pass

# Network:
# 784 Input Neurons 
# 10 Hidden Neurons 
# 10 Output Neurons 

def ReLu(Weighted_Sum):
    return np.maximum(0,Weighted_Sum)

def one_hot(anwser_index):
    one_hot_vector = np.zeros((10, 1))
    one_hot_vector[anwser_index] = 1
    return one_hot_vector

def loss(predictions,one_hot_vector):
    loss_per_neuron = (predictions - one_hot_vector) ** 2
    return np.sum(loss_per_neuron)

nn = NeuralNetwork()

nn.train()
