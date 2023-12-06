import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom as dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Some constants 
INPUT_FOLDER = './data/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()


# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)


first_patient = load_scan(INPUT_FOLDER + patients[0])
first_patient_pixels = get_pixels_hu(first_patient)
plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()

# Show some slice in the middle
plt.imshow(first_patient_pixels[80], cmap=plt.cm.gray)
plt.show()

def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    print("thick:", scan[0].SliceThickness)
    print("spacing: ",scan[0].PixelSpacing[0])
    spacing = np.array(scan[0].SliceThickness + scan[0].PixelSpacing[0], dtype=np.float32)
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing

pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1,1,1])
print("Shape before resampling\t", first_patient_pixels.shape)
print("Shape after resampling\t", pix_resampled.shape)

def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    verts, faces, normals, values  = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()
    


# plot_3d(pix_resampled, 400)


# lung segmentation
def max_label_volume(img, bg=-1):
    vals, counts = np.unique(img, return_counts=True)
    
    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None
    

def seg_lung(img, fill_lung=True):
    binary_img = np.array(img > -320, dtype=np.int8) + 1
    labels = measure.label(binary_img)

    background_label = labels[0,0,0]

    binary_img[background_label == labels] = 2

    if fill_lung:
        for i, axial_slices in enumerate(binary_img):
            axial_slices = axial_slices - 1
            labeling = measure.label(axial_slices)
            lung_max = max_label_volume(labeling, bg=0)

            if lung_max is not None:
                binary_img[i][labeling != lung_max] = 1

    binary_img -= 1
    binary_img = 1 - binary_img

    labels = measure.label(binary_img, background=0)
    lung_max = max_label_volume(labels, bg=0)

    if lung_max is not None:
        binary_img[labels != lung_max] = 0

    return binary_img


segmented_lung = seg_lung(pix_resampled, False)
segmented_lung_fill = seg_lung(pix_resampled, True)

# plot_3d(segmented_lung, 0)
# plot_3d(segmented_lung_fill, 0)
plot_3d(segmented_lung_fill - segmented_lung, 0)
