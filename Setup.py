#%% Get 3 points between start and end point along border of image object
# Input: start point, end point, image object
# Output: 3 points between start and end point along border of image object
#%%
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
#%%
path_to_all_nifti = "data/aortaDataOG"

for idx, file in enumerate(os.listdir(path_to_all_nifti)):

    # Get the path to the folder containing the DICOM files

    path_to_nifti = os.path.join(path_to_all_nifti, file)
    # Get the path to the folder where the NIfTI files will be saved
    s = ''.join(x for x in file if x.isdigit())
    path_to_save = os.path.join('data/dataOrigin', str(s))
    print(path_to_save)

    # # # Convert the DICOM files to NIfTI files
    image = sitk.ReadImage(path_to_nifti)

    image.SetOrigin([0, 0, 0])
    image.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    sitk.WriteImage(image, path_to_save + '.nii.gz')

