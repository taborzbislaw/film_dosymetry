import numpy as np
import json
import sys
import os
import time
import SimpleITK as sitk
from scipy.optimize import minimize
#from pycimg import CImg
#from tqdm import tqdm
import math

def read_json(fname):
    # Load data from JSON file
    with open(fname, 'r') as f:
        data = json.load(f)
    return data

def rational_func(x, a, b, c):
    return (a + b*x) / (c + x)

def inverse_rational_func(x, a, b, c):
    # given channel value x calculates dose
    return (a - c*x) / (x - b) 
    
def omega(opt_densities):
    def f(x):
        a,b,c = calibrationCoefficients[0]
        r = -np.log(rational_func(x, a, b, c))
        delta_r = opt_densities[0]/r

        a,b,c = calibrationCoefficients[1]
        g = -np.log(rational_func(x, a, b, c))
        delta_g = opt_densities[1]/g

        a,b,c = calibrationCoefficients[2]
        b = -np.log(rational_func(x, a, b, c))
        delta_b = opt_densities[2]/b

        return (delta_r-delta_g)**2 + (delta_r-delta_b)**2 + (delta_b-delta_g)**2

    return f

NUM = int(sys.argv[1])
START = int(sys.argv[2])
END = int(sys.argv[3])

COL_START = int(sys.argv[4])
COL_END = int(sys.argv[5])

TOL = float(sys.argv[6])
MAX_ITER = int(sys.argv[7])
NORM_FACTOR = float(sys.argv[8])
DOSE_MAX = float(sys.argv[9])

workDir = os.path.dirname(__file__)

fname = workDir + '/Resources/foo.nii.gz'
imgSITK = sitk.ReadImage(fname)
img = sitk.GetArrayFromImage(imgSITK)

fname = workDir + '/Resources/coefs.json'
coefs = read_json(fname)
calibrationCoefficients = [coefs['parameters']['red'],coefs['parameters']['green'],coefs['parameters']['blue']]

###################################
start_time = time.time()

calibrated_image = np.zeros(img.shape[0:2],dtype=np.float32)
for row in range(START,END):
    for column in range(COL_START,COL_END):

        if column%100==0:
            f = open(workDir + '/Resources/log' + str(NUM) + '.txt','w')
            print(row,START,END,file=f)
            f.close()
            
        values = img[row,column]/NORM_FACTOR
        opt_densities = -np.log(values)
        
        function_to_minimize = omega(opt_densities)

        a = -DOSE_MAX
        b = DOSE_MAX
        k=(math.sqrt(5)-1)/2
        xL=b-k*(b-a)
        xR=a+k*(b-a)
        numIter = 0
        while (b-a)>TOL:
            if function_to_minimize(xL)<function_to_minimize(xR):
                b=xR
                xR=xL
                xL=b-k*(b-a)
            else:
                a=xL
                xL=xR
                xR=a+k*(b-a)
            numIter += 1
            if numIter > MAX_ITER:
                break
                
        calibrated_image[row,column] = (a+b)/2


end_time = time.time()
execution_time = end_time - start_time
print(f'execution time {execution_time} seconds')
###################################

imgOptSITK = sitk.GetImageFromArray(calibrated_image)
fname = workDir + '/Resources/foo' + str(NUM) + '.nii.gz'
sitk.WriteImage(imgOptSITK,fname)

