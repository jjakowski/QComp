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
import numpy as np
print("\n".join(sys.path))


#
#from  qcpy import  *
from qchemistry.qcpy import  *

exec(open('LiH.py').read())



#E_FCI          =  -7.8823494040135165
#E_FCI          =  -7.882121735916801
#E_FCI      (fc) =  -7.882121735916803
#E_FCI          =  -7.881095658328087
#E_FCI      (fc) =  -7.881095658328088
#E_FCI          =  -7.876469456831681
#E_FCI      (fc) =  -7.876469456831681

refdata=   np.array( [ -7.8823494040135165, -7.882121735916801 , -7.881095658328087 ,  -7.876469456831681 ] );

print("REFERENCE                       : ", refdata);
