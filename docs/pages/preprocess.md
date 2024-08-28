The Unispectral SDK provide some efficient and useful preprocess algorithms.
You can download the unispectral sdk [v1.0.1 beta](https://github.com/Unispectral-SW/monarch-preprocess-app-docs/releases/download/unispectral_sdk_v1.0.1/unispectral-1.0.1-cp38-cp38-win_amd64.whl) to experience the following algorithms.

Download the [data.zip](https://github.com/Unispectral-SW/monarch-preprocess-app-docs/releases/download/unispectral_sdk_v1.0.1/data.zip) and unzip it to the path 'your_pip_path\Lib\site-packages\unispectral\datasets' to use the example data.

#### 1. Msc
Multiple scatter correction of spectral curve and image.

3.1 Msc for spectral curve and image.

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

3.2 P-Msc for spectral curve and image.

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

#### 2. Normalization
Normalization of object image by reference.

4.1 Normalization by big reference.

4.1.1 Normalization by white and dark reference with the same exposure time.

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

4.1.2 Normalization by white and dark reference with different exposure times.

```python
from unispectral.datasets.spectral_cube import get_exposure
exposure_obj = get_exposure("path_root", "cube_name")
exposure_ref = get_exposure("path_root", "cube_name")
Xt = Normalization.pixel_normalizer_darkref_exposure(X_obj, X_ref, X_dark, exposure_obj, exposure_ref)
show_imgs([X_obj[:, :, 0], Xt[:, :, 0]], titles=['original', 'normalization by big ref'])
```

4.2 Normalization by small reference.

4.2.1 Normalization by white and dark reference with the same exposure time.
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

4.2.2 Normalization by white and dark reference with different exposure times.

```python
from unispectral.datasets.spectral_cube import get_exposure
exposure_obj = get_exposure("path_root", "cube_name")
exposure_ref = get_exposure("path_root", "cube_name")
Xt = Normalization.mean_normalizer_darkref_exposure(X_obj, X_ref, X_dark, exposure_obj, exposure_ref, ref_roi)
show_imgs([X_obj[:, :, 0], Xt[:, :, 0]], titles=['original', 'normalization by small ref'])
```

#### 3. Registration
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
#### 4. Roi
Use Roi to get region of interest of the image.

6.1 Roi of rectangle.

6.1.1 Get roi of a rectangle by two points.

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

6.1.2 Get roi of rectangle by x, y, w, h

```python
xywh = (318, 300, 300, 500)
X_obj_roi = Roi.get_roi_of_rect(X_obj, mod='xywh', xywh=xywh)
X_obj_roi_mask = Roi.get_roi_mask_of_rect(X_obj, mod='xywh', xywh=xywh).astype(float)
X_obj_mask = Roi.get_img_mask_of_rect(X_obj, mod='xywh', xywh=xywh)
titles = ['original', 'roi', 'roi_mask', 'obj_mask']
show_imgs([X_obj[:, :, 0], X_obj_roi[:, :, 0], X_obj_roi_mask, X_obj_mask[:, :, 0]], titles=titles)
```
> <img src="images/preprocess/roi_rec_xywh.png" width="550" height="350">

6.2 Roi of circle.

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

#### 5. Segmentation
Segmentation of an image.

Select appropriate segmentation algorithm from ['threshold_local', 'threshold_otsu, 'threshold_yen', 'threshold_triangle'].
```python
X, _, _ = load_obj_and_ref_example()
Xt = Segmentation.thresh_segmenters(X, fun='threshold_local', block_size=1111)
show_imgs([X[:, :, 5], Xt[:, :, 5]], titles=['original', 'segmentation'])
```
> <img src="images/preprocess/segmentation.png" width="450" height="300">

#### 6. SNV
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

#### 7. PCA
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

#### 8. Uniformity Correction
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

