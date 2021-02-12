import numpy as np

class XOR:
    def __init__(self, input, hidden, output,learning_rate, epochs):
        self.input = input
        self.hidden = hidden
        self.output = output
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.hidden_weights = np.random.uniform(size=(self.input, self.hidden))
        self.output_weights = np.random.uniform(size=(self.hidden, self.output))
        self.hidden_bias = np.random.uniform(size=(1, self.hidden))
        self.output_bias = np.random.uniform(size=(1, self.output))


    def sigmoid(self,x):
        return (1 / (1 + np.exp(-x)))

    def sigmoid_derivative(self,y):
        return (y * (1 - y))

    def forward_propagation(self, X):
        hidden_layer_sum = np.dot(X, self.hidden_weights) + self.hidden_bias
        hidden_layer_out = self.sigmoid(hidden_layer_sum)
        output_layer_sum = np.dot(hidden_layer_out, self.output_weights) + self.output_bias
        predicted_y = self.sigmoid(output_layer_sum)
        return predicted_y, hidden_layer_out

    def back_propagation(self,X, predicted_y, y, hidden_layer_out):
        #hidden layer
        dloss_dw_h = X.T.dot((predicted_y - y) * self.sigmoid_derivative(predicted_y).dot(
            self.output_weights.T) * self.sigmoid_derivative(hidden_layer_out))
        dloss_dbias_h = (predicted_y - y) * self.sigmoid_derivative(predicted_y).dot(
            self.output_weights.T) * self.sigmoid_derivative(hidden_layer_out)
        # output layer
        dloss_dw_out = hidden_layer_out.T.dot((predicted_y - y) * self.sigmoid_derivative(predicted_y))
        dloss_dbias_out = (predicted_y - y) * self.sigmoid_derivative(predicted_y)
        return dloss_dw_h, dloss_dbias_h,dloss_dw_out,dloss_dbias_out

    def update_weights(self, dloss_dw_h, dloss_dbias_h,dloss_dw_out,dloss_dbias_out):
        #hidden layer
        self.hidden_weights -= dloss_dw_h * self.learning_rate
        self.hidden_bias -= np.sum(dloss_dbias_h, axis=0, keepdims=True)* self.learning_rate
        # output layer
        self.output_weights -= dloss_dw_out * self.learning_rate
        self.output_bias -= np.sum(dloss_dbias_out, axis=0, keepdims=True) * self.learning_rate

    def fit(self, X, y):
        y = y.reshape(y.shape[0], 1)
        for _ in range(self.epochs):
            predicted_y, hidden_layer_out = self.forward_propagation(X)
            dloss_dw_h, dloss_dbias_h,dloss_dw_out,dloss_dbias_out = self.back_propagation(X, predicted_y, y, hidden_layer_out)
            self.update_weights(dloss_dw_h, dloss_dbias_h,dloss_dw_out,dloss_dbias_out)

    def predict(self, X):
        predicted_y, _ = self.forward_propagation(X)
        predictions = (predicted_y >= 0.5)*1
        return (predictions)

