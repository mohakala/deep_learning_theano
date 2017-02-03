import numpy as np
import theano
import theano.tensor as T
from theano import function
from theano import shared

# http://www.analyticsvidhya.com/blog/2016/04/neural-networkds-python-theano/  # export THEANO_FLAGS='device=cpu,floatX=float32'    
# export OMP_NUM_THREADS=4
# srun -n 1 -p gpu --gres=gpu:1 -t 04:00:00 --mem-per-cpu=4G --pty $SHELL

# Variables
a = T.scalar('a')
b = T.scalar('b')
avec = T.vector('av')
bvec = T.vector('bv')



# Scalar expression
c=a*b
f=function([a,b],c)

inp=[1.5,3]
output=f(inp[0],inp[1])
print('Output:',output)

# Shared variables == Constants
sh = shared(0.33)

# Vector expression
cvec=avec*bvec*sh  # element-wise multiplication x sh
fvec=function([avec,bvec],cvec)
in1=[1,2]
in2=[0,10]
output2=fvec(in1,in2)
print('Output2:',output2)

# Expression
x = T.iscalar('x')
sh = shared(0)
f=function([x],sh**2, updates=[(sh,sh+x)])

input=1
for i in range(3):
    print('x=1: output3, sh:',f(input),sh.get_value())





print('Done')
assert False, "stop" 






