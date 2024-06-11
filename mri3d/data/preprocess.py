# split the dataset into patches
from empatches import EMPatches
emp = EMPatches()
patches, indices  = emp.extract_patches(sphere, patchsize=8, overlap=0.0, stride=None, vox=True)