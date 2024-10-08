import numpy as np
import os
import SimpleITK as sitk
import sys
import pymedphys

print(pymedphys.__version__)

workDir = os.path.dirname(__file__)
fname = workDir + '/Resources/ref.nii.gz'
ref = sitk.ReadImage(fname)
spacing = ref.GetSpacing()
origin = ref.GetOrigin()
ref = sitk.GetArrayFromImage(ref)

fname = workDir + '/Resources/eval.nii.gz'
ev = sitk.ReadImage(fname)
ev = sitk.GetArrayFromImage(ev)

gridx = np.arange(ref.shape[0])*abs(spacing[0])
gridy = np.arange(ref.shape[1])*abs(spacing[1])
grid = (gridx,gridy)

doseTol = float(sys.argv[1])
distTol = float(sys.argv[2])
lower_percent_dose_cutoff=float(sys.argv[3])

max_gamma = 2.0
interp_fraction=5

g = pymedphys.gamma(grid,ref,grid,ev,doseTol,distTol,max_gamma=max_gamma,interp_fraction=interp_fraction,lower_percent_dose_cutoff=lower_percent_dose_cutoff)
g = np.nan_to_num(g, nan=0)

g = sitk.GetImageFromArray(g)
g.SetSpacing(spacing)
g.SetOrigin(origin)
fname = workDir + '/Resources/gamma.nii.gz'
sitk.WriteImage(g,fname)


