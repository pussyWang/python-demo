import numpy
import scipy.special


class neuralNetwork:

    def __init__(self, inputnodes, hiddennodes,outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lrate = learningrate
        #self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
        #self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)
        ## 采用正态分布来初始化链接权重
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs_list, target_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        target = numpy.array(target_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        outputs_error = target - final_outputs
        hidden_errors = numpy.dot(self.who, outputs_error)
        self.who += self.lrate * numpy.dot((outputs_error * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        self.wih ++ self.lrate * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

        pass

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
    pass


input_nodes = 3
hidden_nodes = 3
output_nodes = 3
learning_rate = 0.5
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

print(n.query([1.0, 0.5, -1.5]))


