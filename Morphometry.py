#%%
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import scipy
import skimage

#%%
N = 6
path_to_n_nifti = os.path.join(os.getcwd(), 'AorticLandmarkSegmentation/spine_localization_dataset/ImagesT/{}.nii'.format(N))
path_to_csv = os.path.join(os.getcwd(), 'AorticLandmarkSegmentation/spine_localization_dataset/aorta_setup/TransformedCor.csv')

# crop image 
f = np.loadtxt(path_to_csv, delimiter=',')
index = np.where(f[:,0] == N)[0][0]

R = [f[index][1],f[index][2],f[index][3]]
L = [f[index][4],f[index][5],f[index][6]]
N = [f[index][7],f[index][8],f[index][9]]
RLC = [f[index][10],f[index][11],f[index][12]]
RNC = [f[index][13],f[index][14],f[index][15]]
LNC = [f[index][16],f[index][17],f[index][18]]

image = sitk.ReadImage(path_to_n_nifti, sitk.sitkFloat32)
print(image.GetOrigin())
print(image.GetDirection())
print(image.GetSpacing())

R = image.TransformPhysicalPointToIndex(R)
L = image.TransformPhysicalPointToIndex(L)
N = image.TransformPhysicalPointToIndex(N)
RLC = image.TransformPhysicalPointToIndex(RLC)
RNC = image.TransformPhysicalPointToIndex(RNC)
LNC = image.TransformPhysicalPointToIndex(LNC)

print(R)

image1 = sitk.GetArrayFromImage(image)

plt.figure()
implot = plt.imshow(image1[:, R[1], :], cmap='gray', origin='lower')
plt.scatter(R[0], R[2], c='r', s=40)
plt.scatter(RLC[0], RLC[2], c='r', s=40)
plt.scatter(RNC[0], RNC[2], c='r', s=40)

plt.figure()
implot = plt.imshow(image1[R[2], :, :], cmap='gray', origin='lower')
plt.scatter(R[0], R[1], c='r', s=40)
# plt.scatter(RLC[0], RLC[2], c='r', s=40)
# plt.scatter(RNC[0], RNC[2], c='r', s=40)

#%%
# function that rotates the 3D sitk image to a specified normal
TestImage = image1[:, R[1], :]
plt.figure()
implot = plt.imshow(TestImage, cmap='gray', origin='lower')
plt.scatter(R[0], R[2], c='r', s=40)
plt.scatter(RLC[0], RLC[2], c='r', s=40)
plt.scatter(RNC[0], RNC[2], c='r', s=40)

average = np.mean([[R[0],R[2]], [RLC[0], RLC[2]], [RNC[0], RNC[2]]], axis=0)
med = TestImage[int(np.rint(average[0])), int(np.rint(average[1]))]

#treshold image
threshold_filter = sitk.BinaryThresholdImageFilter()
threshold_filter.SetLowerThreshold(400)
threshold_filter.SetUpperThreshold(600)
threshold_filter.SetInsideValue(1)
threshold_filter.SetOutsideValue(0)
binary_image = threshold_filter.Execute(image)



plt.figure()
implot = plt.imshow(sitk.GetArrayFromImage(binary_image)[:, R[1], :], cmap='gray', origin='lower')
plt.scatter(R[0], R[2], c='r', s=40)
plt.scatter(RLC[0], RLC[2], c='r', s=40)
plt.scatter(RNC[0], RNC[2], c='r', s=40)

filter_dilation = sitk.BinaryDilateImageFilter()
filter_dilation.SetKernelRadius(1)
binary_image = filter_dilation.Execute(binary_image)



# # points = connect_border_points(image, [[R[0], R[2]], [RLC[0], RLC[2]], [RNC[0], RNC[2]]])

# edges = sitk.CannyEdgeDetection(image, lowerThreshold=100, upperThreshold=130)


# plt.figure()
# implot = plt.imshow(sitk.GetArrayFromImage(edges)[:, R[1], :], cmap='gray', origin='lower')
# plt.scatter(R[0], R[2], c='r', s=40)
# plt.scatter(RLC[0], RLC[2], c='r', s=40)
# plt.scatter(RNC[0], RNC[2], c='r', s=40)

# plt.figure()
# implot = plt.imshow(sitk.GetArrayFromImage(binary_image)[:, R[1], :], cmap='gray', origin='lower')
# plt.scatter(R[0], R[2], c='r', s=40)
# plt.scatter(RLC[0], RLC[2], c='r', s=40)
# plt.scatter(RNC[0], RNC[2], c='r', s=40)

#%%

# Convert the binary image to a distance map
distance_map = sitk.SignedMaurerDistanceMap(binary_image)

closetP = distance_map
map = sitk.GetArrayFromImage(closetP)[:, R[1], :]

plt.figure()
implot = plt.imshow(map, cmap='gray', origin='lower')
plt.scatter(R[0], R[2], c='r', s=40)
plt.scatter(RLC[0], RLC[2], c='r', s=40)
plt.scatter(RNC[0], RNC[2], c='r', s=40)

print(np.max(map), np.min(map))
test_border = map == 0

plt.figure()
implot = plt.imshow(test_border, cmap='gray', origin='lower')
#%% cv2 find contour test
binary = sitk.GetArrayFromImage(binary_image)[:, R[1], :]
plt.figure()
implot = plt.imshow(binary, cmap='gray', origin='lower')
plt.scatter(R[0], R[2], c='r', s=40)
plt.scatter(RLC[0], RLC[2], c='r', s=40)
plt.scatter(RNC[0], RNC[2], c='r', s=40)

filter_contour  = sitk.BinaryContourImageFilter()
contour = filter_contour.Execute(binary_image)
plt.figure()
implot = plt.imshow(sitk.GetArrayFromImage(contour)[:, R[1], :], cmap='gray', origin='lower')   
plt.scatter(R[0], R[2], c='r', s=40)
plt.scatter(RLC[0], RLC[2], c='r', s=40)
plt.scatter(RNC[0], RNC[2], c='r', s=40)
contour = sitk.GetArrayFromImage(contour)[:, R[1], :]
#%% find closest point on contour to R, RLC and RNC


contour_points = np.argwhere(contour == 1)
contour_points = np.flip(contour_points, axis=1)
plt.figure()
implot = plt.imshow(contour, cmap='gray', origin='lower')
plt.scatter(contour_points[:, 0], contour_points[:, 1], c='g', s=1)

# find closest point on contour to R
pointRNC = [RNC[0], RNC[2]]
pointRLC = [RLC[0], RLC[2]]
pointR = [R[0], R[2]]
dist1 = np.linalg.norm(contour_points - pointR, axis=1)
closest_point_start = contour_points[np.argmin(dist1)]

# find closest point on contour to RLC
dist2 = np.linalg.norm(contour_points - pointRLC, axis=1)
closest_point_end = contour_points[np.argmin(dist2)]

plt.figure()
implot = plt.imshow(contour, cmap='gray', origin='lower')
plt.scatter(R[0], R[2], c='r', s=40)
plt.scatter(RLC[0], RLC[2], c='r', s=40)
plt.scatter(RNC[0], RNC[2], c='r', s=40)
plt.scatter(closest_point_start[0], closest_point_start[1], c='g', s=40)

# Walk from R to RLC and save points to array
points = []
points.append(closest_point_start)
print(points[-1])

idx = 0
while(points[-1] != closest_point_end).all():
    idx += 1
    # find neighobouring points on contour around last point in points
    neighbours = contour_points[np.linalg.norm(contour_points - points[-1], axis=1) < 3]
    # remove points that are already in points
    # neighbours = neighbours[~np.isin(neighbours, points).all(axis=1)]
    # find closest to end point
    dist = np.linalg.norm(neighbours - closest_point_end, axis=1)
    closest_point = neighbours[np.argmin(dist)]
    points.append(closest_point)
    # print("last point ", points[-1])
    # print("closest point ", closest_point)
    # print("neighbours ", neighbours)

    print(idx)
    if idx > 100:
        break

    
# delaaaa
points = np.array(points)
plt.figure()
implot = plt.imshow(contour, cmap='gray', origin='lower')
plt.scatter(points[:, 0], points[:, 1], c='g', s=3)
plt.scatter(R[0], R[2], c='r', s=40)
plt.scatter(RLC[0], RLC[2], c='r', s=40)

#%%
















