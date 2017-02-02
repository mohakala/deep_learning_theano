import numpy as np
import theano
import theano.tensor as T
from theano import function
from theano import shared

# http://www.analyticsvidhya.com/blog/2016/04/neural-networkds-python-theano/  # THEANO_FLAGS='device=cpu,floatX=float32'    


# Variables
a = T.scalar('a')
b = T.scalar('b')
avec = T.vector('av')
bvec = T.vector('bv')

# Shared variables == Constants
sh = shared(0.33)


# Scalar expression
c=a*b
f=function([a,b],c)

# Vector expression
cvec=avec*bvec*sh  # element-wise multiplication x sh
fvec=function([avec,bvec],cvec)


# Evaluate expression
inp=[1.5,3]
output=f(inp[0],inp[1])
print('Output:',output)


# Evaluate vector expresssion
in1=[1,2]
in2=[0,10]
output2=fvec(in1,in2)
print('Output2:',output2)











assert True==False, "stop" 



print('Done')




