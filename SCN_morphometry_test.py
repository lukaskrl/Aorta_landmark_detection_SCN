#%% Import packages
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from pytest import skip
from requests import get
from CNNTesting.dataset import *
from CNNTesting.parts import *
import math
#%% functions

def getContourPoints(point_start, point_end, contour_array, search_radious = 10):
    output_points = np.array([point_start])
    idx = 0
    while (output_points[-1] != point_end).any():
        # get last point
        point_last = output_points[-1]
        # get contour points in radious
        contour_points_in_radious = []
        for point in contour_array:
            if math.dist(point,point_last) < search_radious:
                contour_points_in_radious.append(point)

                
        # get closest point to next input point
        point_next = point_last
        min_distance = 100000
        for point in contour_points_in_radious:
            if math.dist(point,point_end) < min_distance:
                min_distance = math.dist(point,point_end)
                point_next = point
        contour_array = np.delete(contour_array, np.where((contour_array == point_next).all(axis=1)), axis=0)
        # add next point to contour_points
        output_points = np.append(output_points, [point_next], axis=0)
        idx += 1
        if idx > 100:
            break
    return output_points

def findClosestPoint(point_start, contour_array):
    min_distance = 100000
    closest_point = []
    for point in contour_array:
        if math.dist(point,point_start) < min_distance:
            min_distance = math.dist(point,point_start)
            closest_point = point
    return closest_point
    

def bresenhamLine(x0, y0, x1, y1):
    # https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    points = []
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    
    err = dx - dy
    
    while True:
        points.append((x0, y0))
        
        if x0 == x1 and y0 == y1:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    
    return points

def getCircumscribedCircle(points):
    # https://stackoverflow.com/questions/20314306/find-arc-circle-equation-given-three-points-in-space-3d

    A = np.linalg.norm(points[1]-points[2])
    B = np.linalg.norm(points[0]-points[2])
    C = np.linalg.norm(points[0]-points[1])
    
    s = (A + B + C) / 2
    radius = A * B * C / 4 / np.sqrt(s * (s - A) * (s - B) * (s - C))
      
    circumference = 2 * np.pi * radius
    
    return circumference 

def rotateImageToPlane(input_image, points, normalize = False, filter = False, sigma=1.0):
    normal = np.cross(points[0]-points[1], points[0]-points[2])
    normal = normal / np.linalg.norm(normal)
    # calculate axis  and angle for versor rotation
    axis = np.cross(normal, np.array([0,1,0]))
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.dot(normal, np.array([0,1,0])))
    # rotate image
    versor = sitk.VersorTransform()
    versor.SetIdentity()
    versor.SetRotation(axis, angle)
    versor.SetCenter(points[0])
    resampled_image = sitk.Resample(input_image, versor.GetInverse(), sitk.sitkLinear, 0.0, input_image.GetPixelID())
    
    # normalize image
    if normalize:
        resampled_image = sitk.Normalize(resampled_image)
        
    if filter:
        resampled_image = sitk.SmoothingRecursiveGaussian(resampled_image, sigma)
    
    transformed_indexs = []
    for point in points:
        trans_point = versor.TransformPoint(point)
        transformed_indexs.append(input_image.TransformPhysicalPointToIndex(trans_point))
        
    return  resampled_image, transformed_indexs, versor
    
def getKernelValuesAroundIndex(image2D, index, windows_size = 3):  
    row, col = index.astype(int)

    # Extract the window around the specified point
    window = image2D[row - windows_size : row + windows_size + 1, col - windows_size : col + windows_size + 1]

    return window
    

def extractIslandAroundIndex(image2D, seed_point, connectivity = 3):
    connected_components = sitk.ConnectedComponent(image, connectivity)

    # Step 2: Create binary mask for the connected component containing the seed point
    seed_label = connected_components[seed_point]
    binary_mask = (connected_components == seed_label)

    # Step 3: Use the binary mask to extract the region of interest
    extracted_region = sitk.Mask(image, binary_mask)

    return extracted_region
  
def getContourLength(input_image, points, show_steps=False):

    resampled_image, transformed_indexs, versor = rotateImageToPlane(input_image, points, normalize=True)
        
    image_slice = sitk.GetArrayFromImage(resampled_image)[:,transformed_indexs[0][1],:]
    itk_image_slice = sitk.GetImageFromArray(image_slice)       
    
    # get mean coordinates of the transformed indexes for binary thresholding
    mean_index = np.round(np.flip(np.mean(transformed_indexs, axis=0)))
    
    # island = extractIslandAroundIndex(itk_image_slice, mean_index)
    # plt.imshow(sitk.GetArrayFromImage(island), cmap='gray')
    # plt.show()
    
    mean_values = getKernelValuesAroundIndex(image_slice, mean_index[[0,2]], windows_size=10)
    
    mean_value = np.median(mean_values)
    max_value = np.max(mean_values)
    min_value = np.min(mean_values)
    
    # upper threshold doesnt matter for aortic valves, the valve is the brightest part of the image
    upper_binary_treshold = max_value*2
    lower_binary_treshold = min_value*1.2
    
    # binary treshold image filter
    binary_filter = sitk.BinaryThresholdImageFilter()
    binary_filter.SetLowerThreshold(int(lower_binary_treshold))
    binary_filter.SetUpperThreshold(int(upper_binary_treshold))
    binary_filter.SetInsideValue(1)
    binary_filter.SetOutsideValue(0)
    binary_image = binary_filter.Execute(itk_image_slice)

    radius = 1
    
    
    
    dilate_filter = sitk.BinaryDilateImageFilter()
    dilate_filter.SetKernelRadius(radius)
    dilate_filter.SetForegroundValue(1)
    dilate_filter.SetBackgroundValue(0)
    dilate_image = dilate_filter.Execute(binary_image)
    
    erode_filter = sitk.BinaryErodeImageFilter()
    erode_filter.SetKernelRadius(radius)
    erode_filter.SetForegroundValue(1)
    erode_filter.SetBackgroundValue(0)
    erode_image = erode_filter.Execute(dilate_image)

    contour_filter = sitk.BinaryContourImageFilter()
    contour_image = contour_filter.Execute(erode_image)
    
    contour_array = np.array(np.where(sitk.GetArrayFromImage(contour_image) == 1)).T
    # switch columns becouse sitk.GerArrayFromImage returns (z,y,x) and we need (x,y,z)
    contour_array = np.flip(contour_array, axis=1)

    transformed_indexs = np.delete(transformed_indexs, 1, axis=1)
    
    contour1 = getContourPoints(findClosestPoint(transformed_indexs[0], contour_array), 
                                findClosestPoint(transformed_indexs[1], contour_array), 
                                contour_array)
    contour2 = getContourPoints(findClosestPoint(transformed_indexs[1], contour_array), 
                                findClosestPoint(transformed_indexs[2], contour_array), 
                                contour_array)

    
    spacing = resampled_image.GetSpacing()
    spacing = np.array([spacing[0], spacing[2]])
    contour1_spaced = contour1 * spacing
    contour2_spaced = contour2 * spacing
    
    # calculate length of both contours
    length1 = 0
    length2 = 0
    for i in range(len(contour1)-1):
        length1 += math.dist(contour1_spaced[i], contour1_spaced[i+1])
    for i in range(len(contour2)-1):
        length2 += math.dist(contour2_spaced[i], contour2_spaced[i+1])
    contour_length = length1 + length2
    
    if show_steps:
        plt.title('image slice')
        plt.imshow(image_slice, cmap='gray')
        plt.scatter(transformed_indexs[0][0], transformed_indexs[0][1], c='b')
        plt.scatter(transformed_indexs[1][0], transformed_indexs[1][1], c='b')
        plt.scatter(transformed_indexs[2][0], transformed_indexs[2][1], c='b')
        # plt.scatter(mean_index[0], mean_index[2], c='r')
        plt.show()
        
        plt.title('Binary image')
        plt.imshow(sitk.GetArrayFromImage(binary_image), cmap='gray')
        plt.show()

        plt.title('erode image')
        plt.imshow(sitk.GetArrayFromImage(erode_image), cmap='gray')
        plt.show()
        
        plt.title('dilate image')
        plt.imshow(sitk.GetArrayFromImage(dilate_image), cmap='gray')
        plt.show()
    
        plt.title('contour image')
        plt.imshow(sitk.GetArrayFromImage(contour_image), cmap='gray')
        plt.scatter(transformed_indexs[0][0], transformed_indexs[0][1], c='b')
        plt.scatter(transformed_indexs[1][0], transformed_indexs[1][1], c='b')
        plt.scatter(transformed_indexs[2][0], transformed_indexs[2][1], c='b')
        plt.show()
        
        plt.title('contours')
        plt.imshow(sitk.GetArrayFromImage(contour_image), cmap='gray')
        plt.scatter(findClosestPoint(transformed_indexs[0], contour_array)[0], findClosestPoint(transformed_indexs[0], contour_array)[1], c='r')
        plt.scatter(findClosestPoint(transformed_indexs[1], contour_array)[0], findClosestPoint(transformed_indexs[1], contour_array)[1], c='r')
        plt.scatter(findClosestPoint(transformed_indexs[2], contour_array)[0], findClosestPoint(transformed_indexs[2], contour_array)[1], c='b')

        
        plt.scatter(contour1[:,0], contour1[:,1], c='r', s=1)
        plt.scatter(contour2[:,0], contour2[:,1], c='b', s=1)
        plt.show()
    return contour_length
     
def getCuspHeightGH(input_image, points, show_steps=False): 

    resampled_image, transformed_indexs, versor = rotateImageToPlane(input_image, points, normalize=True, filter = True, sigma=0.5)
   
    image_slice = sitk.GetArrayFromImage(resampled_image)[:,transformed_indexs[0][1],:]
    # image_slice in z x space
    
    itk_image_slice = sitk.GetImageFromArray(image_slice) 
    
    # itk back to x z space
    mean_index = np.flip(transformed_indexs[1])
    transformed_indexs = np.delete(transformed_indexs, 1, axis=1)

    # slice in z x space, but index now also fliped to z x
    mean_values = getKernelValuesAroundIndex(image_slice, mean_index[[0,2]], windows_size=10)    
    
    # plt.imshow(mean_values, cmap='gray')
    # plt.show()
    mean_value = np.median(mean_values)
    max_value = np.max(mean_values)
    min_value = np.min(mean_values)
    
    upper_binary_treshold = max_value*2
    lower_binary_treshold = min_value*1
    
    # binary treshold image filter
    binary_filter = sitk.BinaryThresholdImageFilter()
    binary_filter.SetLowerThreshold(lower_binary_treshold)
    binary_filter.SetUpperThreshold(upper_binary_treshold)
    binary_filter.SetInsideValue(1)
    binary_filter.SetOutsideValue(0)
    binary_image = binary_filter.Execute(itk_image_slice)

    radius = 1
    erode_filter = sitk.BinaryErodeImageFilter()
    erode_filter.SetKernelRadius(radius)
    erode_filter.SetForegroundValue(1)
    erode_filter.SetBackgroundValue(0)
    erode_image = erode_filter.Execute(binary_image)
 
    dilate_filter = sitk.BinaryDilateImageFilter()
    dilate_filter.SetKernelRadius(radius*2)
    dilate_filter.SetForegroundValue(1)
    dilate_filter.SetBackgroundValue(0)
    dilate_image = dilate_filter.Execute(erode_image)
    
    erode_filter.SetKernelRadius(radius)
    erode_filter.SetForegroundValue(1)
    erode_filter.SetBackgroundValue(0)
    erode_image = erode_filter.Execute(dilate_image)
    
    # binaryFillHoleFilter = sitk.BinaryFillholeImageFilter()
    # binaryFillHoleFilter.SetForegroundValue(1)
    # erode_image = binaryFillHoleFilter.Execute(binary_image)
    
    
    filtered_image_array = sitk.GetArrayFromImage(erode_image)
    # image now in z x space from normal sitk.GetArrayFromImage
    
    # find closest point all points that lay on the line between point 2 and 3
    bresenham_line = bresenhamLine(transformed_indexs[1][0], transformed_indexs[1][1], transformed_indexs[2][0], transformed_indexs[2][1])
    bresenham_line = np.array(bresenham_line)   

    # find the closest point in the bresenhamLine to the point[1], that is allso appart of the contour.
    mid_cusp_insertion_point = []
    test_line_point = []
    min_distance = 100000
    for point in bresenham_line:
        # turn points around since sitk.array flips arrays
        # check neoughbouring points to point in filtered_image_array, if any has value == 0
        
        
        if (filtered_image_array[point[1], point[0]] == 0):
            test_line_point.append(point)
            if math.dist(point, transformed_indexs[1]) < min_distance:
                min_distance = math.dist(point, transformed_indexs[1])
                mid_cusp_insertion_point = point
       
    if show_steps:
        plt.title('image slice')
        plt.imshow(image_slice, cmap='gray')
        plt.scatter(transformed_indexs[0][0], transformed_indexs[0][1], c='b')
        plt.scatter(transformed_indexs[1][0], transformed_indexs[1][1], c='b')
        plt.scatter(transformed_indexs[2][0], transformed_indexs[2][1], c='b')
        plt.scatter(bresenham_line[:,0], bresenham_line[:,1], c='r', s=1)
        # plt.scatter(mean_index[0], mean_index[2], c='r')
        plt.show()
        
        plt.title('Binary image')
        plt.imshow(sitk.GetArrayFromImage(binary_image), cmap='gray')
        plt.show()

        plt.title('dilate image')
        plt.imshow(sitk.GetArrayFromImage(dilate_image), cmap='gray')
        plt.show() 
        
        plt.title('erode image')
        plt.imshow(sitk.GetArrayFromImage(erode_image), cmap='gray')
        plt.show() 
                
    # insertion point is in x z space, TODO neki narobe
    mid_cusp_insertion_point = [int(mid_cusp_insertion_point[0]), int(mean_index[1]), int(mid_cusp_insertion_point[1])]
    mid_cusp_insertion_point_world = input_image.TransformIndexToPhysicalPoint(mid_cusp_insertion_point)
    
    # transform point back before rotation
    mid_cusp_insertion_point_world = versor.GetInverse().TransformPoint(mid_cusp_insertion_point_world)
    
    # calculate distance from mid_cusp_insertion_point to transformed_indexs[0]
    cusp_height = math.dist(mid_cusp_insertion_point_world, points[0])
    test_height = math.dist(mid_cusp_insertion_point_world, points[2])

    if show_steps:
        
        plt.title('final image points')
        plt.imshow(image_slice, cmap='gray')
        plt.scatter(transformed_indexs[0][0], transformed_indexs[0][1], c='r')
        plt.scatter(transformed_indexs[1][0], transformed_indexs[1][1], c='r')
        plt.scatter(transformed_indexs[2][0], transformed_indexs[2][1], c='r')
        plt.scatter(bresenham_line[:,0], bresenham_line[:,1], c='r', s=1)
        plt.scatter(mid_cusp_insertion_point[0], mid_cusp_insertion_point[2], c='b') 
        plt.show()

        
    return cusp_height, mid_cusp_insertion_point_world
    # return 0, 0
        
def getCuspHeightEH(points, mid_cusp_point):
    # calculate plane norm of points
    normal = np.cross(points[0]-points[1], points[0]-points[2])
    normal = normal / np.linalg.norm(normal)
    
    # calculate distance from mid_cusp_point to plane
    cusp_height = abs(np.dot(normal, mid_cusp_point - points[0]))
    return cusp_height

    
def getMorphometry(input_image, points):
    R = points[0,:]
    L = points[1,:]
    N = points[2,:]

    RLC = points[3,:]
    RNC = points[4,:]
    LNC = points[5,:]

    meanCommisure = (RLC + RNC + LNC) / 3
    meanCusp = (R + L + N) / 3
    
    contourLengthL = getContourLength(input_image, [RLC,L,LNC], show_steps=False)
    contourLengthN = getContourLength(input_image, [LNC,N,RNC], show_steps=False)
    contourLengthR = getContourLength(input_image, [RNC,R,RLC], show_steps=False)

    cuspHeightGHL, mid_pointL = getCuspHeightGH(input_image, [L,meanCommisure,meanCusp], show_steps=False)
    cuspHeightGHN, mid_pointN = getCuspHeightGH(input_image, [R,meanCommisure,meanCusp], show_steps=False)
    cuspHeightGHR, mid_pointR = getCuspHeightGH(input_image, [N,meanCommisure,meanCusp], show_steps=False)

    cuspHeightEHL = getCuspHeightEH([L,R,N], mid_pointL)
    cuspHeightEHR = getCuspHeightEH([R,L,N], mid_pointR)
    cuspHeightEHN = getCuspHeightEH([N,R,L], mid_pointN)

    Radius = getCircumscribedCircle([R,L,N])
    
    return contourLengthL, contourLengthN, contourLengthR, cuspHeightGHL, cuspHeightGHN, cuspHeightGHR, cuspHeightEHL, cuspHeightEHN, cuspHeightEHR, Radius


#%% Load image and landmark test
# load the model
idx = 10
# model = SCN(in_channels=1, num_classes=6)
# model.load_state_dict(torch.load('/root/models/13-11-2023_18-16/model130.ckpt'))

# load image and points
pathToImage = "/root/data/dataOrigin/"+str(idx)+".nii.gz"
pathToPoints = "/root/data/LandmarkCoordinates.csv"
image = sitk.ReadImage(pathToImage)   

# load points
landmarks_frame = pd.read_csv(pathToPoints, header=None)
landmarks_array = landmarks_frame.to_numpy()
index_points = np.where(landmarks_array[:, 0] == idx)
landmarks_array = landmarks_array[index_points, :].squeeze()  
landmarks_array = landmarks_array[1:]
landmarks_array = landmarks_array.reshape(6, 3)

landmark_index = []
for i in range(len(landmarks_array)):
    point = landmarks_array[i,:]
    landmark_index.append(
        image.TransformPhysicalPointToIndex(point))

point = 2
plt.imshow(sitk.GetArrayFromImage(image)[:,landmark_index[point][1],:], cmap='gray')
plt.scatter(landmark_index[point][0], landmark_index[point][2], c='r')


#%% Pipeline for morphometry extraction 1 image
R = landmarks_array[0,:]
L = landmarks_array[1,:]
N = landmarks_array[2,:]

RLC = landmarks_array[3,:]
RNC = landmarks_array[4,:]
LNC = landmarks_array[5,:]

meanCommisure = (RLC + RNC + LNC) / 3
meanCusp = (R + L + N) / 3

contourLengthL = getContourLength(image, [RLC,L,LNC], show_steps=False)
contourLengthN = getContourLength(image, [LNC,N,RNC], show_steps=False)
contourLengthR = getContourLength(image, [RNC,R,RLC], show_steps=False)

cuspHeightGHL, mid_pointL = getCuspHeightGH(image, [L,meanCommisure,meanCusp], show_steps=True)
cuspHeightGHN, mid_pointN = getCuspHeightGH(image, [R,meanCommisure,meanCusp], show_steps=True)
cuspHeightGHR, mid_pointR = getCuspHeightGH(image, [N,meanCommisure,meanCusp], show_steps=True)


cuspHeightEHL = getCuspHeightEH([L,R,N], mid_pointL)
cuspHeightEHR = getCuspHeightEH([R,L,N], mid_pointR)
cuspHeightEHN = getCuspHeightEH([N,R,L], mid_pointN)

R = getCircumscribedCircle([R,L,N])

print('Contour length L: {:.2f}'.format(contourLengthL))
print('Contour length N: {:.2f}'.format(contourLengthN))
print('Contour length R: {:.2f}'.format(contourLengthR))
print('Cusp height GH L: {:.2f}'.format(cuspHeightGHL))
print('Cusp height GH N: {:.2f}'.format(cuspHeightGHN))
print('Cusp height GH R: {:.2f}'.format(cuspHeightGHR))
print('Cusp height EH L: {:.2f}'.format(cuspHeightEHL))
print('Cusp height EH N: {:.2f}'.format(cuspHeightEHN))
print('Cusp height EH R: {:.2f}'.format(cuspHeightEHR))

print('bazal ring circumference: {:.2f}'.format(R))

#%% iterate through foldrer and save results to csv
# create csv
pathToImagesFolder = "/root/data/dataOrigin/"
pathToLandMarks = "/root/data/LandmarkCoordinates.csv"
morphometry_array = pd.DataFrame(columns=['id', 'Contour length L', 'Contour length N', 'Contour length R', 'Cusp height GH L', 'Cusp height GH N', 'Cusp height GH R', 'Cusp height EH L', 'Cusp height EH N', 'Cusp height EH R', 'bazal ring circumference'])
landmarks_array = pd.read_csv(pathToLandMarks, header=None)
landmarks_array = landmarks_array.to_numpy()

for i in range(landmarks_array.shape[0]):
    index = int(landmarks_array[i,0])
    print(i, index)
    if index > 125:
        continue
    image = sitk.ReadImage(pathToImagesFolder+str(index)+'.nii.gz')
    landmarks = landmarks_array[i,1:].reshape(6,3)
    morphometry = getMorphometry(image, landmarks)
    
    # save to csv
    morphometry_array.loc[i] = [index, morphometry[0], morphometry[1], morphometry[2], morphometry[3], morphometry[4], morphometry[5], morphometry[6], morphometry[7], morphometry[8], morphometry[9]]

    
#%%
print(morphometry_array)