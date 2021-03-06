import numpy as np, theano, theano.tensor as T,pickle,random

class Network(object):

  def __init__(self,sizes,activation=T.nnet.selu):
    self.sizes = sizes
    self.b = [theano.shared(np.random.randn(i)) for i in self.sizes[1:]]
    self.w = [theano.shared(nextlayer ** -.5 * np.random.randn(nextlayer,current)) for current,nextlayer in zip(self.sizes[:-1],self.sizes[1:])]
    self.activation = activation

    self.x = T.vector("x")
    self.y = T.vector("y")
    self.lr = T.scalar()
    
    self.model = [self.activation((T.dot(self.w[0],self.x)) + self.b[0])]
    for i in range(1,len(self.sizes)-2):
      self.model.append(self.activation((T.dot(self.w[i],self.model[-1])) + self.b[i]))
    self.model.append(T.nnet.sigmoid((T.dot(self.w[-1],self.model[-1])) + self.b[-1]))

    self.out = self.model[-1]
    self.feedForward = theano.function([self.x],self.out,name="FF")

    
    self.cost = T.sum((self.out-self.y)**2)
    self.updB = T.grad(self.cost,self.b)
    self.updW = T.grad(self.cost,self.w)
    
    self.upd = theano.function([self.x,self.y,self.lr],self.cost,updates=
      [(self.b[i],self.b[i]-self.updB[i]*self.lr) for i in range(len(self.b))]  +
      [(self.w[i],self.w[i]-self.updW[i]*self.lr) for i in range(len(self.w))],name="train")
    
    self.ev = theano.function([self.x,self.y],self.cost)

    #Needed to implement Batch/Minibatch training in the future
    
    self.xs = T.matrix("xs")
    self.ys = T.matrix("ys")

    
  def strain(self,data,epochs,lr): #Stochastic training for n epochs
    for i in range(epochs):
        for x,y in data:
            self.upd(x,y,lr)
                
  def evaluate(self,data):
    out =(sum([self.ev(x,y) for x,y in data])/len(data))
    return out

  def list(self):
    for k in self.__dict__:
      print(k,"=",self.__dict__[k],"\nType:",type(self.__dict__[k]),"\n")

  def save(self,fname):
    d = self.__dict__.copy()
    pickle.dump(d,open(fname,"wb"))
  
  def load(self,fname):
    d = pickle.load(open(fname,"rb"))
    for k in d:
      self.__dict__[k] = d[k]
