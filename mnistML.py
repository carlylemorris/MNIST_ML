import numpy as np,theano,mnist_loader,Network,numpy as np,time

tdata, vdata, test = mnist_loader.load_data_wrapper() #Stole from Neural Networks and Deep Learning book

network = Network.Network([784,200,200,10])

for epoch in range(10):
    c = 0
    for x,y in vdata:
        if np.argmax(network.feedForward(x)) == np.argmax(y):
            c = c + 1
    print("\nEpoch",epoch,"\nTrain Cost:",network.evaluate(tdata),"\nValidation Cost:",network.evaluate(vdata),"\nValidation % Accuracy:",c*100/len(vdata))
    network.strain(tdata,1,.001)

c = 0
for x,y in vdata:
    if np.argmax(network.feedForward(x)) == np.argmax(y):
        c = c + 1
print("\nFinal test % accuracy:",c*100/len(test))    
