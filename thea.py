import numpy as np
import theano
import theano.tensor as T
from theano import function
from theano import shared

"""
http://www.analyticsvidhya.com/blog/2016/04/neural-networkds-python-theano/  # export THEANO_FLAGS='device=cpu,floatX=float32'    

export OMP_NUM_THREADS=4
srun -n 1 -p gpu --gres=gpu:1 -t 04:00:00 --mem-per-cpu=4G --pty $SHELL

installed TDM-GCC for C-implementation
installed Theano and libpython
 http://www.lfd.uci.edu/~gohlke/pythonlibs/
 pip install --no-index --find-links C:\Download Theano (& libpython) 
"""

print('**Theano examples')

# Variables
a = T.scalar('a')
b = T.scalar('b')
avec = T.vector('av')
bvec = T.vector('bv')



# Evaluate scalar expression
print('Evaluate scalar expression')
c=a*b
f=function([a,b],c)

inp=[1.5,3]
output=f(inp[0],inp[1])
print('Output:',output)



# Evaluate vector expression
# shared variables == Constants
print('Evaluate vector expression')
sh = shared(0.33)
cvec=avec*bvec*sh  # element-wise multiplication x sh
fvec=function([avec,bvec],cvec)

in1=[1,2]
in2=[0,10]
output2=fvec(in1,in2)
print('Output2:',output2)


# Evaluate expression multiple times
print('Evaluate expression multiple times')
x = T.iscalar('x')
sh = shared(0)
f=function([x],sh**2, updates=[(sh,sh+x)])
input=1
for i in range(3):
    print('x=1: output3, sh:',f(input),sh.get_value())


# *Theano functions
# Function returns multiple values
print('Function returns multiple values')
a = T.dscalar('a')
f = function([a],[a**2, a**3])
print(f(3))



# Function returns the gradient
print('Function returns gradient')
x = T.dscalar('a')
y=x**4
dy=T.grad(y,x)
f = function([x],dy)
print(f(3))



#from theano import pp  #pretty-print
#print(pp(qy))




# *Single neuron - Feed forward
print('**Single neuron - feed forward')
from theano.ifelse import ifelse

#Define variables:
x = T.vector('x')
w = theano.shared(np.array([1,1]))
b = theano.shared(-1.5)


#Evaluate expression
z = T.dot(x,w)+b
a = ifelse(T.lt(z,0),0,1)
neuron = theano.function([x],a)


#Define inputs and weights
inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

#Iterate through all inputs and find outputs:'
print('- iterated through inputs and find outputs')
for i in range(len(inputs)):
    t = inputs[i]
    out = neuron(t)
    print ('- the output for x1=%d | x2=%d is %d' % (t[0],t[1],out) )


# *Single neuron - Backward propagation, loss, training - AND gate
print('**Single neuron - Backward propagatin, loss, training, AND gate')
from random import random

#Define variables:
x = T.matrix('x')
w = theano.shared(np.array([random(),random()]))
b = theano.shared(1.)
learning_rate = 0.01


#Define mathematical expression:
z = T.dot(x,w)+b
a = 1/(1+T.exp(-z))


#Full-batch gradient descent

#Cost: simple logistic cost for classification
a_hat = T.vector('a_hat') #Actual output
cost = -(a_hat*T.log(a) + (1-a_hat)*T.log(1-a)).sum()


#Gradients and how to update the weights
dw,db = T.grad(cost,[w,b])

train = function(
    inputs = [x,a_hat],
    outputs = [a,cost],
    updates = [
        [w, w-learning_rate*dw],
        [b, b-learning_rate*db]
    ]
)



#Define inputs and weights
inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
outputs = [0,0,0,1]

#Iterate through all inputs and find outputs:
print('- iterate through inputs and find outputs')
cost = []
predList = []
for iteration in range(300):  # Orig: 30000
    pred, cost_iter = train(inputs, outputs)
    cost.append(cost_iter)
    predList.append([pred[0],pred[3]])
    
#Print the outputs:
print ('The outputs of the NN are:')
for i in range(len(inputs)):
    print ('The output for x1=%d | x2=%d is %.2f' % (inputs[i][0],inputs[i][1],pred[i]) )



    
#Plot the flow of cost:
print ('\nThe flow of cost during model run is as following:')
import matplotlib.pyplot as plt
# %matplotlib inline
plt.plot(cost,'-')
plt.plot(predList,'--')
plt.show()



# *Two neurons - XOR-gate
print('**Two neurons - XOR gate')
import theano
import theano.tensor as T
from theano.ifelse import ifelse
import numpy as np
from random import random

#Define variables:
x = T.matrix('x')
w1 = theano.shared(np.array([random(),random()]))
w2 = theano.shared(np.array([random(),random()]))
w3 = theano.shared(np.array([random(),random()]))
b1 = theano.shared(1.)
b2 = theano.shared(1.)
learning_rate = 0.01

#Define mathematical expressions
a1 = 1/(1+T.exp(-T.dot(x,w1)-b1))
a2 = 1/(1+T.exp(-T.dot(x,w2)-b1))
x2 = T.stack([a1,a2],axis=1)
a3 = 1/(1+T.exp(-T.dot(x2,w3)-b2))

#Gradient and update rule
a_hat = T.vector('a_hat') #Actual output
cost = -(a_hat*T.log(a3) + (1-a_hat)*T.log(1-a3)).sum()
dw1,dw2,dw3,db1,db2 = T.grad(cost,[w1,w2,w3,b1,b2])

train = function(
    inputs = [x,a_hat],
    outputs = [a3,cost],
    updates = [
        [w1, w1-learning_rate*dw1],
        [w2, w2-learning_rate*dw2],
        [w3, w3-learning_rate*dw3],
        [b1, b1-learning_rate*db1],
        [b2, b2-learning_rate*db2]
    ]
)

#Train the model
inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
outputs = [1,0,0,1]

#Iterate through all inputs and find outputs:
cost = []
for iteration in range(30000):
    pred, cost_iter = train(inputs, outputs)
    cost.append(cost_iter)
    
#Print the outputs:
print ('The outputs of the NN are:')
for i in range(len(inputs)):
    print ('The output for x1=%d | x2=%d is %.2f' % (inputs[i][0],inputs[i][1],pred[i]))
    
#Plot the flow of cost:
print ('\nThe flow of cost during model run is as following:')
import matplotlib.pyplot as plt
plt.plot(cost)
plt.show()




assert False, "-- FORCED STOP --"



print('Done')
 






