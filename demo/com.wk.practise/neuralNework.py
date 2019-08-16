import numpy


class neuralNetwork:

    def __init__(self, inputnodes, hiddennodes,outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lrate = learningrate
        #self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
        #self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)
        ## 采用正态分布来初始化链接权重
        self.wih = (numpy.random.normal(0.0, pow(self.hnodes, -0.5)),
                    (self.hnodes, self.inodes))
        self.who = (numpy.random.normal(0.0, pow(self.onodes, -0.5)),
                    (self.onodes, self.hnodes))
        pass

    def train(self):
        pass

    def query(self):
        hidden_inputs = numpy.dot(self.wih, input_nodes)
        hidden_outputs = self
        pass


    pass


input_nodes=3
hidden_nodes=3
output_nodes=3
learning_rate=0.5



n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

init_weight=numpy.random.rand(input_nodes, hidden_nodes) - 0.5

wih = numpy.random.normal(0.0, pow(hidden_nodes, -0.5),
            (hidden_nodes, input_nodes))
who = numpy.random.normal(0.0, pow(output_nodes, -0.5),
            (output_nodes, hidden_nodes))
print (wih)

print ("###############")

print(who)


