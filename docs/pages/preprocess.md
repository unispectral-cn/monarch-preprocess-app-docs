The Unispectral SDK provide some efficient and useful preprocess algorithms.
You can download the unispectral sdk [v1.0.1 beta](https://github.com/Unispectral-SW/monarch-preprocess-app-docs/releases/download/unispectral_sdk_v1.0.1/unispectral-1.0.1-cp38-cp38-win_amd64.whl) to experience the following algorithms.

Download the [data.zip](https://github.com/Unispectral-SW/monarch-preprocess-app-docs/releases/download/unispectral_sdk_v1.0.1/data.zip) and unzip it to the path 'your_pip_path\Lib\site-packages\unispectral\datasets' to use the example data.


#### 1. Uniformity Correction
Using this UC algorithm can effectively eliminate angular shift issue in the images.
```python
from unispectral.preprocessing.uniformity_correction import UCModelInfer
from unssolomon.UnsSolomonAPI import *

#init camera
camera = UnsSolomonController()
result = camera.InitCam(device_id=0, unsSensorMode=UnsSernsortMode.emUnsMode_ALL)
print("init camera result", result)
time.sleep(15)

uc_model_path=r'path/to/model.h5'
wanted_cwls=[510,530,550,570,590,610,630,650,670,690,713,736,759,782,805,828,851,874,897,920]

#get the supported bands of this camera
bands_valid = camera.GetAvailableBands()

#load the UC model
uc_model_infer = UCModelInfer(uc_model_path)

#get the capture plan.
cap_cwls=uc_model_infer.get_captured_cwls(wanted_cwls,bands_valid,"ultra")

#validate the cap_cwls
cap_cwls=np.intersect1d(cap_cwls,bands_valid)

#capture the original data
[rawData, rgbData] = camera.CaptureCustomLUT(cap_cwls)
cap_cube=np.array(rawData)

#use UC model to get the corrected data
corr_wanted_cube = uc_model_infer.infer(cap_cube,cap_cwls,wanted_cwls)

```

#### 2. Msc
Multiple scatter correction of spectral curve and image.

2.1 Msc for spectral curve and image.

```python
from unispectral.datasets import load_obj_and_ref_example
from unispectral.preprocessing.msc import Msc
from unispectral.preprocessing.segmentation import Segmentation
from unispectral.visualization.utils import show_imgs, show_curves_random
from unispectral.datasets.spectral_cube import load_cube
import numpy as np
# load X_obj by your own path or uns example
# X_obj = load_cube("path_obj").data
X_obj, _, _ = load_obj_and_ref_example()
# background segmentation
X = Segmentation.thresh_segmenters(X_obj, fun='threshold_local')
# spectral curve
X_spc = X.reshape(-1, 10)
# remove background
X_spc = X_spc[X_spc[:, 0] != 0, :]
# mean spectral curve
X_mean = np.mean(X_spc, axis=0)

# msc of spc
Xt = Msc.msc(X_spc, X_mean=X_mean)
show_curves_random([X_spc, Xt], titles=['before msc', 'after msc'])

# msc of img
Xt = Msc.msc(X, X_mean=X_mean)
show_imgs([X[:, :, 5], Xt[:, :, 5]], titles=['before msc', 'after msc'])

```
> <img src="images/preprocess/msc_spc.png" width="450" height="300">

> <img src="images/preprocess/msc_img.png" width="450" height="300">

2.2 P-Msc for spectral curve and image.

```python
# p-msc of spc that means partition msc by setting p which is the number of partition.
Xt = Msc.pmsc(X_spc, X_mean=X_mean, p=2)
show_curves_random([X_spc, Xt], titles=['before p-msc', 'after p-msc'])

# p-msc of img that means partition msc by setting p which is the number of partition.
Xt = Msc.pmsc(X, X_mean=X_mean, p=2)
show_imgs([X[:, :, 5], Xt[:, :, 5]], titles=['before p-msc', 'after p-msc'])

```
> <img src="images/preprocess/p-msc_spc.png" width="450" height="300">

> <img src="images/preprocess/p-msc_img.png" width="450" height="300">

#### 3. Normalization
Normalization of object image by reference.

3.1 Normalization by big reference.

3.1.1 Normalization by white and dark reference with the same exposure time.

```python
from unispectral.visualization.utils import show_imgs
from unispectral.preprocessing.normalization import Normalization
from unispectral.datasets import load_obj_and_ref_example
# TODO Load the cube of object, reference and dark.
X_obj, X_ref, X_dark = load_obj_and_ref_example()
Xt = Normalization.pixel_normalizer(X_obj, X_ref)
show_imgs([X_obj[:, :, 0], Xt[:, :, 0]], titles=['original', 'normalization by big ref'])
Xt = Normalization.pixel_normalizer_darkref(X_obj, X_ref, X_dark)
show_imgs([X_obj[:, :, 0], Xt[:, :, 0]], titles=['original', 'normalization by big ref'])

```
> <img src="images/preprocess/normalization_by_big_ref-1.png" width="450" height="300">

> <img src="images/preprocess/normalization_by_big_ref-2.png" width="450" height="300">

3.1.2 Normalization by white and dark reference with different exposure times.

```python
from unispectral.datasets.spectral_cube import get_exposure
exposure_obj = get_exposure("path_root", "cube_name")
exposure_ref = get_exposure("path_root", "cube_name")
Xt = Normalization.pixel_normalizer_darkref_exposure(X_obj, X_ref, X_dark, exposure_obj, exposure_ref)
show_imgs([X_obj[:, :, 0], Xt[:, :, 0]], titles=['original', 'normalization by big ref'])
```

3.2 Normalization by small reference.

3.2.1 Normalization by white and dark reference with the same exposure time.
```python
from unispectral.visualization.utils import show_imgs
from unispectral.preprocessing.normalization import Normalization

ref_roi = (300, 300, 350, 350)
Xt = Normalization.mean_normalizer(X_obj, X_ref, ref_roi)
show_imgs([X_obj[:, :, 0], Xt[:, :, 0]], titles=['original', 'normalization by small ref'])
Xt = Normalization.mean_normalizer_darkref(X_obj, X_ref, X_dark, ref_roi)
show_imgs([X_obj[:, :, 0], Xt[:, :, 0]], titles=['original', 'normalization by small ref'])
```
> <img src="images/preprocess/normalization_by_small_ref-1.png" width="450" height="300">

> <img src="images/preprocess/normalization_by_small_ref-2.png" width="450" height="300">

3.2.2 Normalization by white and dark reference with different exposure times.

```python
from unispectral.datasets.spectral_cube import get_exposure
exposure_obj = get_exposure("path_root", "cube_name")
exposure_ref = get_exposure("path_root", "cube_name")
Xt = Normalization.mean_normalizer_darkref_exposure(X_obj, X_ref, X_dark, exposure_obj, exposure_ref, ref_roi)
show_imgs([X_obj[:, :, 0], Xt[:, :, 0]], titles=['original', 'normalization by small ref'])
```

#### 4. Registration
Registration of images of all bands.
```python
from unispectral.datasets.spectral_cube import load_cube
from unispectral.preprocessing.registration import Registration
from unispectral.visualization.utils import show_imgs
import numpy as np
X = load_cube("path_obj").data
Xt = Registration.fast_registration(X)
show_imgs([np.sum(X, axis=2), np.sum(Xt, axis=2)], titles=['original', 'registration'])
```
> <img src="images/preprocess/registration.png" width="450" height="300">

#### 5. Roi
Use Roi to get region of interest of the image.

5.1 Roi of rectangle.

5.1.1 Get roi of a rectangle by two points.

```python
from unispectral.datasets import load_obj_and_ref_example
from unispectral.preprocessing.roi import Roi
X_obj, _, _ = load_obj_and_ref_example()

point1 = (318, 300)
point2 = (618, 800)
X_obj_roi = Roi.get_roi_of_rect(X_obj, point1, point2)
X_obj_roi_mask = Roi.get_roi_mask_of_rect(X_obj, point1, point2).astype(float)
X_obj_mask = Roi.get_img_mask_of_rect(X_obj, point1, point2)
titles = ['original', 'roi', 'roi_mask', 'obj_mask']
show_imgs([X_obj[:, :, 0], X_obj_roi[:, :, 0], X_obj_roi_mask, X_obj_mask[:, :, 0]], titles=titles)
```
> <img src="images/preprocess/roi_rec_point.png" width="550" height="350">

5.1.2 Get roi of rectangle by x, y, w, h

```python
xywh = (318, 300, 300, 500)
X_obj_roi = Roi.get_roi_of_rect(X_obj, mod='xywh', xywh=xywh)
X_obj_roi_mask = Roi.get_roi_mask_of_rect(X_obj, mod='xywh', xywh=xywh).astype(float)
X_obj_mask = Roi.get_img_mask_of_rect(X_obj, mod='xywh', xywh=xywh)
titles = ['original', 'roi', 'roi_mask', 'obj_mask']
show_imgs([X_obj[:, :, 0], X_obj_roi[:, :, 0], X_obj_roi_mask, X_obj_mask[:, :, 0]], titles=titles)
```
> <img src="images/preprocess/roi_rec_xywh.png" width="550" height="350">

5.2 Roi of circle.

Get roi of a circle by center points and radius.

```python
center = (453, 667)
radius = 300
X_obj_roi = Roi.get_roi_of_rect(X_obj, mod='circle', center=center, radius=radius)
X_obj_roi_mask = Roi.get_roi_mask_of_rect(X_obj, mod='circle', center=center, radius=radius)
X_obj_mask = Roi.get_img_mask_of_rect(X_obj, mod='circle', center=center, radius=radius)
titles = ['original', 'roi', 'roi_mask', 'obj_mask']
show_imgs([X_obj[:, :, 0], X_obj_roi[:, :, 0], X_obj_roi_mask, X_obj_mask[:, :, 0]], titles=titles)
```
> <img src="images/preprocess/roi_circle_center.png" width="550" height="350">

#### 6. Segmentation
Segmentation of an image.

Select appropriate segmentation algorithm from ['threshold_local', 'threshold_otsu, 'threshold_yen', 'threshold_triangle'].
```python
X, _, _ = load_obj_and_ref_example()
Xt = Segmentation.thresh_segmenters(X, fun='threshold_local', block_size=1111)
show_imgs([X[:, :, 5], Xt[:, :, 5]], titles=['original', 'segmentation'])
```
> <img src="images/preprocess/segmentation.png" width="450" height="300">

#### 7. SNV
Standard Normal Variate transformation of spectral curve.

The input reflectivity X will be transformed to absorption by log10(X).

```python
from pylab import log10
from unispectral.preprocessing.snv import SNV
from unispectral.visualization.utils import show_curves
from unispectral.datasets import load_curves_example
X, y = load_curves_example()
snv = SNV()
Xt = snv.snv_spc(X)
show_curves([log10(1.0 / X), Xt], legend=['original', 'snv'])
```
> <img src="images/preprocess/snv.png" width="450" height="300">

#### 8. PCA
Principal component analysis (PCA).
```python
from unispectral.datasets import load_curves_example
from unispectral.preprocessing.pca import Pca
import matplotlib.pyplot as plt
X, y = load_curves_example()
Xt = Pca.pca(X, n_components=2)
plt.figure()
plt.scatter(Xt[y==1, 0], Xt[y==1, 1], c='blue', label='peanut_type_1')
plt.scatter(Xt[y==2, 0], Xt[y==2, 1], c='green', label='peanut_type_2')
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.legend()
plt.show()
```
> <img src="images/preprocess/pca.png" width="450" height="300">

#### 9. MoveAverage
Move average.
```python
import spectral
import numpy as np
import os
import matplotlib.pyplot as plt
from unispectral.preprocessing.move_average import MoveAverage

def read_spectral_curve_from_file(dir_name, file_name, roi, radius):
    FIX_DARK = 64
    hdr_path = os.path.join(dir_name, file_name + ".hdr")
    raw_path = os.path.join(dir_name, file_name + ".raw")
    cube_array = spectral.envi.open(hdr_path, raw_path).load(dtype=np.uint16).asarray().copy()
    X = cube_array - FIX_DARK
    spc = np.mean(X[roi[1] - radius: roi[1] + radius, roi[0] - radius: roi[0] + radius, :], axis=(0, 1))
    return spc

def show_move_average(bands, spc_ref, spc_ma, bands_ma):
    print(spc_ma, bands_ma)
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Spectral Curve Move Average')
    plt.xlim([713, 920])
    ax1.set_xlim([713, 920])
    ax1.grid()
    ax2.grid()
    ax1.set_xticks(bands)
    ax2.set_xticks(bands)
    ax2.set_xlim([713, 920])
    ax1.plot(bands, spc_ref)
    ax2.plot(bands_ma, spc_ma)
    plt.show()

filename = "ENVI_cube_20230928_101700"
cube_dir = r"C:\Users\uns_n\Documents\SpecCurves\spec_curves_772_20230928_101653\cube_20230928_101700"
bands = np.array([713, 736, 759, 782, 805, 828, 851, 874, 897, 920])
roi_ref = (614, 512)
roi_obj = 666, 512
radius = 18 // 2
average_step = 3

spc_ref = read_spectral_curve_from_file(cube_dir, filename, roi_ref, radius)
spc_ma = MoveAverage.moving_average(spc_ref, average_step)
bands_ma = MoveAverage.moving_average(bands, average_step)
show_move_average(bands, spc_ref, spc_ma, bands_ma)
```
> <img src="images/preprocess/move_average.png" width="450" height="300">

#### 10. Vector Normalization
Vector Normalization.
```python
import numpy as np
from sklearn.decomposition import PCA
import spectral
from sklearn.preprocessing import StandardScaler
import os
import math
import matplotlib.pyplot as plt
from unispectral.preprocessing.vector_normalization import VectorNormalization


def read_spectral_curve_from_file(dir_name, file_name, roi, radius):
    FIX_DARK = 64
    hdr_path = os.path.join(dir_name, file_name + ".hdr")
    raw_path = os.path.join(dir_name, file_name + ".raw")
    cube_array = spectral.envi.open(hdr_path, raw_path).load(dtype=np.uint16).asarray().copy()
    X = cube_array - FIX_DARK
    spc = np.mean(X[roi[1] - radius: roi[1] + radius, roi[0] - radius: roi[0] + radius, :], axis=(0, 1))
    return spc


def show_vec_normal(bands, spc_ref, bands_ma, spc_ma):
    print(spc_ma, bands_ma)
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Spectral Curve Vector Normalization')
    plt.xlim([713, 920])
    ax1.set_xlim([713, 920])
    ax1.grid()
    ax2.grid()
    ax1.set_xticks(bands)
    ax2.set_xticks(bands)
    ax2.set_xlim([713, 920])
    for i in range(len(spc_ref)):
        ax1.plot(bands, spc_ref[i])
    for i in range(len(spc_ma)):
        ax2.plot(bands_ma, spc_ma[i])

    plt.show()


dirname = "cube_20230928_101700"
cube_dir = r"C:\Users\uns_n\Documents\SpecCurves\spec_curves_772_20230928_101653"

bands = np.array([713, 736, 759, 782, 805, 828, 851, 874, 897, 920])
roi_ref = (614, 512)
roi_obj = 666, 512
radius = 18 // 2
X_spc = []

for filename in os.listdir(cube_dir):
    f = os.path.join(cube_dir, filename)
    if os.path.isdir(f):
        spc = read_spectral_curve_from_file(cube_dir + "\\" + filename, "ENVI_" + filename, roi_obj, radius)
        X_spc.append(spc)

X_spc_vn = VectorNormalization.vector_normalization(X_spc)
show_vec_normal(bands, X_spc, bands, X_spc_vn)
```
> <img src="images/preprocess/vec_normal.png" width="450" height="300">

#### 11. Discrete Fourier Transform
Discrete Fourier Transform.
```python
import spectral
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from unispectral.preprocessing.fourier_transform import FourierTransform

def show_fourier(r_list, i_list):
    idx = [i for i in range(1, len(r_list) + 1)]
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Fourier Transform')

    ax1.grid()
    ax2.grid()

    ax1.stem(idx, r_list, use_line_collection=True, basefmt="none")
    ax1.set_xticks(idx)
    ax1.set_xlabel("real")

    ax2.stem(idx, i_list, use_line_collection=True, basefmt="none")
    ax2.set_xticks(idx)
    ax2.set_xlabel("imaginary")

    plt.show()


dirname = "cube_20230928_101700"
cube_dir = r"C:\Users\uns_n\Documents\SpecCurves\spec_curves_772_20230928_101653\cube_20230928_101700"
FIX_DARK = 64
hdr_path = os.path.join(cube_dir, "ENVI_" + dirname + ".hdr")
raw_path = os.path.join(cube_dir, "ENVI_" + dirname + ".raw")
cube_array = spectral.envi.open(hdr_path, raw_path).load(dtype=np.uint16).asarray().copy()
bands = np.array([713, 736, 759, 782, 805, 828, 851, 874, 897, 920])
roi_ref = 614, 512
roi_obj = 666, 512
radius = 18 // 2

X = cube_array - FIX_DARK
spc_ref = np.mean(X[roi_ref[1] - radius: roi_ref[1] + radius, roi_ref[0] - radius: roi_ref[0] + radius, :], axis=(0, 1))

r_list, i_list = FourierTransform.fourier_transform(spc_ref)
show_fourier(r_list, i_list)
```
> <img src="images/preprocess/fourier_transform.png" width="450" height="300">

#### 12. Mean Centering
Mean Centering.
```python
import numpy as np
import spectral
import os
import math
import matplotlib.pyplot as plt
from unispectral.preprocessing.mean_centering import MeanCentering

def show_mean_centering(bands, spc_ref, spc_ma):
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Spectral Curve Mean Centering')
    plt.xlim([713, 920])
    ax1.set_xlim([713, 920])
    ax1.grid()
    ax2.grid()
    ax1.set_xticks(bands)
    ax2.set_xticks(bands)
    ax2.set_xlim([713, 920])
    for i in range(len(spc_ref)):
        ax1.plot(bands, spc_ref[i])
    for i in range(len(spc_ma)):
        ax2.plot(bands, spc_ma[i])

    plt.show()

def read_spectral_curve_from_file(dir_name, file_name, roi, radius):
    FIX_DARK = 64
    hdr_path = os.path.join(dir_name, file_name + ".hdr")
    raw_path = os.path.join(dir_name, file_name + ".raw")
    cube_array = spectral.envi.open(hdr_path, raw_path).load(dtype=np.uint16).asarray().copy()
    X = cube_array - FIX_DARK
    spc = np.mean(X[roi[1] - radius: roi[1] + radius, roi[0] - radius: roi[0] + radius, :], axis=(0, 1))
    return spc


dirname = "cube_20230928_101700"
cube_dir = r"C:\Users\uns_n\Documents\SpecCurves\spec_curves_772_20230928_101653"

bands = np.array([713, 736, 759, 782, 805, 828, 851, 874, 897, 920])
roi_ref = (614, 512)
roi_obj = 666, 512
radius = 18 // 2

X_spc = []
for filename in os.listdir(cube_dir):
    f = os.path.join(cube_dir, filename)
    if os.path.isdir(f):
        spc = read_spectral_curve_from_file(cube_dir + "\\" + filename, "ENVI_" + filename, roi_obj, radius)
        X_spc.append(spc)


X_mean_center = MeanCentering.mean_centering(X_spc)
show_mean_centering(bands, X_spc, X_mean_center)
```
> <img src="images/preprocess/mean_centering.png" width="450" height="300">

#### 13. Correction
Correction of images of all bands.
```python
from unispectral.datasets.spectral_cube import load_cube
from unispectral.preprocessing.correction import Correction
from unispectral.visualization.utils import show_imgs
import numpy as np
cube_name = r"E:\data\hyper_spectral\sample\01\state8050_20231218_174639"
X_obj = load_cube(cube_name).data
path_model_type = './test_correction/pkl/peakW/peakW_test.mat'
bands_hyper = list(range(713, 921, 5))
bands = list(range(713, 921, 5))
kernel_size = 4
X_obj_correction = Correction.cwl_angle_corrector(X_obj, bands_hyper, bands, path_model_type, kernel_size)
show_imgs([X_obj[:, :, 16], X_obj_correction[:, :, 16]], titles=['before correction', 'after correction'])
```
> <img src="images/preprocess/correction_angular.png" width="450" height="300">

#### 14. Autoscaling
Autoscaling.
```python
import numpy as np
import spectral
import os
import matplotlib.pyplot as plt
from unispectral.preprocessing.autoscaling import Autoscaling


def read_spectral_curve_from_file(dir_name, file_name, roi, radius):
    FIX_DARK = 64
    hdr_path = os.path.join(dir_name, file_name + ".hdr")
    raw_path = os.path.join(dir_name, file_name + ".raw")
    cube_array = spectral.envi.open(hdr_path, raw_path).load(dtype=np.uint16).asarray().copy()
    X = cube_array - FIX_DARK
    spc = np.mean(X[roi[1] - radius: roi[1] + radius, roi[0] - radius: roi[0] + radius, :], axis=(0, 1))
    return spc


def show_autoscaling(bands, spc_ref, col_mean, col_std):
    # print(spc_ma, bands_ma)
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Spectral Curve Autoscaling')
    plt.xlim([713, 920])
    ax1.set_xlim([713, 920])
    ax1.grid()
    ax2.grid()
    ax1.set_xticks(bands)
    ax2.set_xticks(bands)
    ax2.set_xlim([713, 920])
    for i in range(len(spc_ref)):
        ax1.plot(bands, spc_ref[i])

    trans_spc = []
    for row_id in range(len(spc_ref)):
        spc_row = []
        for col_id in range(len(spc_ref[0])):
            trans = (spc_ref[row_id][col_id] - col_mean[col_id]) / col_std[col_id]
            spc_row.append(trans)
        trans_spc.append(spc_row)

    for i in range(len(spc_ref)):
        ax2.plot(bands, trans_spc[i])

    plt.show()


dirname = "cube_20230928_101700"
cube_dir = r"C:\Users\uns_n\Documents\SpecCurves\spec_curves_772_20230928_101653\cube_20230928_101700"
cubes_dir = r"C:\Users\uns_n\Documents\SpecCurves\spec_curves_772_20230928_101653"

bands = np.array([713, 736, 759, 782, 805, 828, 851, 874, 897, 920])
roi_ref = (614, 512)
roi_obj = 666, 512
radius = 18 // 2

X_spc = []
for filename in os.listdir(cubes_dir):
    f = os.path.join(cubes_dir, filename)
    if os.path.isdir(f):
        spc = read_spectral_curve_from_file(cubes_dir + "\\" + filename, "ENVI_" + filename, roi_obj, radius)
        X_spc.append(spc)


col_mean, col_std = Autoscaling.autoscaling(X_spc)
show_autoscaling(bands, X_spc, col_mean, col_std)
```
> <img src="images/preprocess/autoscaling.png" width="450" height="300">

#### 15. Difference First
Autoscaling.
```python
import spectral
import numpy as np
import os
import matplotlib.pyplot as plt
from unispectral.preprocessing.difference_first import DifferenceFirst


def read_spectral_curve_from_file(dir_name, file_name, roi, radius):
    FIX_DARK = 64
    hdr_path = os.path.join(dir_name, file_name + ".hdr")
    raw_path = os.path.join(dir_name, file_name + ".raw")
    cube_array = spectral.envi.open(hdr_path, raw_path).load(dtype=np.uint16).asarray().copy()
    X = cube_array - FIX_DARK
    spc = np.mean(X[roi[1] - radius: roi[1] + radius, roi[0] - radius: roi[0] + radius, :], axis=(0, 1))
    return spc


def diff_order_bands(bands, diff_order):
    band_list = []
    for idx in range(len(bands) - diff_order * 2):
        band_list.append(bands[idx + diff_order])
    return band_list


def show_difference_order(bands, spc_ref, diff_bands, spc_diff):
    # print(spc_ma, bands_ma)
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Spectral Curve Difference Order')
    plt.xlim([713, 920])
    ax1.set_xlim([713, 920])
    ax1.set_ylabel("spec")
    ax1.grid()
    ax2.grid()
    ax1.set_xticks(bands)
    ax2.set_xticks(bands)
    ax2.set_xlim([713, 920])
    ax2.set_xlabel("wave length")
    for i in range(len(spc_ref)):
        ax1.plot(bands, spc_ref[i])

    for i in range(len(spc_ref)):
        ax2.plot(diff_bands, spc_diff[i])

    plt.show()


filename = "ENVI_cube_20230928_101700"
cubes_dir = r"C:\Users\uns_n\Documents\SpecCurves\spec_curves_772_20230928_101653"

bands = np.array([713, 736, 759, 782, 805, 828, 851, 874, 897, 920])
roi_ref = (614, 512)
roi_obj = 666, 512
radius = 18 // 2

X_spc = []
for filename in os.listdir(cubes_dir):
    f = os.path.join(cubes_dir, filename)
    if os.path.isdir(f):
        spc = read_spectral_curve_from_file(cubes_dir + "\\" + filename, "ENVI_" + filename, roi_obj, radius)
        X_spc.append(spc)

diff_order = 1
spc_diff = DifferenceFirst.difference_first(X_spc, diff_order)
diff_bands = diff_order_bands(bands, diff_order)

show_difference_order(bands, X_spc, diff_bands, spc_diff)
```
> <img src="images/preprocess/difference_first.png" width="450" height="300">

