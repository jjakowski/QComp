#--  debug  function 
#import imp
#import mymodule
# myfun(...)
#imp.reload(mymodule)
# myfun(...)
#mymodule.__file__       # check if file
#--


# export PYTHONPATH=/Users/j2c/Dropbox-ORNL/P3HT-transport/H2-test
# export  PYTHONPATH=/Users/j2c/Dropbox-ORNL/Python-codes



#
#  check  search  path for python:
import sys
print("\n".join(sys.path))


#
#from  qcpy import  *
from qchemistry.qcpy import  *

exec(open('HH.py').read())


refdata=[-1.13414767 ,  -0.5971778 ,  -0.5971778 ,  -0.5971778 ,    0.35228457] ;
print("REFERENCE   : ", refdata);
