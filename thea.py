import numpy as np
import theano
import theano.tensor as T
from theano import function
from theano import shared

# http://www.analyticsvidhya.com/blog/2016/04/neural-networkds-python-theano/  # export THEANO_FLAGS='device=cpu,floatX=float32'    
#
# export OMP_NUM_THREADS=4
# srun -n 1 -p gpu --gres=gpu:1 -t 04:00:00 --mem-per-cpu=4G --pty $SHELL
#
# http://www.lfd.uci.edu/~gohlke/pythonlibs/
# pip install --no-index --find-links C:\Download Theano


# Variables
a = T.scalar('a')
b = T.scalar('b')
avec = T.vector('av')
bvec = T.vector('bv')



# Evaluate scalar expression
c=a*b
f=function([a,b],c)

inp=[1.5,3]
output=f(inp[0],inp[1])
print('Output:',output)



# Evaluate vector expression
# shared variables == Constants
sh = shared(0.33)
cvec=avec*bvec*sh  # element-wise multiplication x sh
fvec=function([avec,bvec],cvec)

in1=[1,2]
in2=[0,10]
output2=fvec(in1,in2)
print('Output2:',output2)


# Evaluate expression multiple times
x = T.iscalar('x')
sh = shared(0)
f=function([x],sh**2, updates=[(sh,sh+x)])
input=1
for i in range(3):
    print('x=1: output3, sh:',f(input),sh.get_value())


# *Theano functions
# Function returns multiple values
a = T.dscalar('a')
f = function([a],[a**2, a**3])
print(f(3))



# Function returns the gradient
x = T.dscalar('a')
y=x**4
dy=T.grad(y,x)
f = function([x],dy)
print(f(3))



#from theano import pp  #pretty-print
#print(pp(qy))



assert False, "-- forced stop --"

# *Single neuron
# Feed forward

from theano.ifelse import ifelse

#Define variables:
x = T.vector('x')
w = T.vector('w')
b = T.scalar('b')

#Define mathematical expression:
z = T.dot(x,w)+b
a = ifelse(T.lt(z,0),0,1)

neuron = theano.function([x,w,b],a)

print('Done')
 






