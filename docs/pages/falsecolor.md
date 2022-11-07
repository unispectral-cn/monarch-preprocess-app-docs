The Unispectral SDK provide the following adapters for developers to analysis spectral cube in a simple and convenient way.
#### 1. PseudoRGBUiAdapter
The pseudo RGB feature allows you to select 3 bands as the pseudo red band, pseudo green band and pseudo blue band and visualize a pseudo RGB false color result. 

```python
from unispectral.false_color import PseudoRGBUiAdapter
import os
import matplotlib.pyplot as plt

adapter = PseudoRGBUiAdapter(image_mode="rgba")
adapter.load_cube("cube_20220826_145339"))
output = adapter.exec([0, 4, 9])

fig, ax = plt.subplots(1, 1)
ax.imshow(output[:, :, :-1])
ax.set_axis_off()
plt.show()
```
> <img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/199901758-38b6bf77-d09b-41f0-9eca-f4923af846c6.png?raw=true" width="320" height="256">

#### 2. CubeImageUiAdapter
By using the cube image feature, you can preview the image of each band and validate the if there is over exposure or under exposure for each band image. The area will be marked in Red if there is over exposure. 

```python
from unispectral.false_color import CubeImageUiAdapter
import matplotlib.pyplot as plt

adapter = CubeImageUiAdapter(cmap="gray", image_mode="argb")
adapter.load_cube("cube_20221009_170210"))
output = adapter.exec(4)

fig, ax = plt.subplots(1, 1)
ax.imshow(output[:, :, 1:])
ax.set_axis_off()
plt.show()

fig, ax = plt.subplots(1, 1)
plt.plot(adapter.hist_)
plt.show()
```
show the over-exposed pixels
> <img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/199905593-ae7e7e98-8295-4fd3-bd54-f49502783d7c.png?raw=true" width="320" height="256">

show histogram of brightness
> <img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/199905815-e10873a3-9873-4d92-a876-727b5140821e.png?raw=true" width="320" height="256">

#### 3. PcaUiAdapter
PCA is the principal component analysis method which can represent a transformation of the original image bands to a set of new, uncorrelated features. By using the PCA, you can inspect the variance of among different components. 

```python
from unispectral.false_color import PcaUiAdapter
import matplotlib.pyplot as plt

adapter = PcaUiAdapter(image_mode="rgba", cache_dir=".cache")
adapter.load_cube("cube_20220826_145339"))

fig, ax = plt.subplots(1, 1)
output = adapter.exec(2)
ax.imshow(output[:, :, :-1])
ax.set_axis_off()
ax.set_title(f"component:{i}")
plt.show()
```

> <img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/199906884-5bdb807d-ce24-42b3-9b53-3bac2b5827ca.png?raw=true" width="320" height="256">

#### 4. KmeansUiAdapter
The k-means algorithm takes an iterative approach to generating clusters. The parameter k specifies the desired number of clusters. K-means algorithm divides image pixels into k groups based on spectral similarity of the pixels without using any prior knowledge of the spectral classes. Each pixel in the image is assigned to the nearest cluster.

```python
from unispectral.false_color import KmeansUiAdapter
import matplotlib.pyplot as plt

nclusters = 3
adapter = KmeansUiAdapter(
    image_mode="rgba", nclusters=nclusters, cache_dir=".cache"
)
adapter.load_cube("cube_20220826_145339"))

output_img = adapter.exec(0)
fig, ax = plt.subplots(1, 1)
ax.imshow(output_img[:, :, :-1])
plt.show()
```

> <img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/199907615-79c8039f-a2e8-4455-a809-39a290d7bb56.png?raw=true" width="320" height="256">

#### 5. StdUiAdapter
Calculate STD of the spectral cube in the spectral dimension and normalize it by the average intensity. The STD reflects the dispersion of the spectral data from the average value. The larger the STD, the greater the data dispersion.

```python
from unispectral.false_color import StdUiAdapter
import matplotlib.pyplot as plt

adapter = StdUiAdapter(image_mode="rgba", cache_dir=".cache")
adapter.load_cube(os.path.join("cube_20220826_145339"))
output = adapter.exec()
print(adapter.min_, adapter.init_max_, adapter.max_)
output = adapter.exec(0, 2.6)
fig, ax = plt.subplots(1, 1)
ax.imshow(output[:, :, :-1])
ax.set_axis_off()
plt.show()
```
> 0.0 8.305647850036621 8.305647850036621

> <img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/199909707-01effaad-9731-4f72-96f5-ef1b0459f6aa.png?raw=true" width="320" height="256">

#### 6. SamUiAdapter
A spectral angle refers to the angle between two spectral curves in N-space. The SAM algorithm uses the spectral angles for classifying data by calculating the smallest angle to the reference spectral curves.

```python
from unispectral.false_color import SamUiAdapter
import matplotlib.pyplot as plt

adapter = SamUiAdapter(
    image_mode="rgba",
    cache_dir=".cache",
    points=[[759, 333], [725, 441], [1000, 328]],
)
adapter.load_cube("cube_20220826_145339"))
output = adapter.exec(0)
fig, ax = plt.subplots(1, 1)
ax.imshow(output[:, :, :-1])
plt.show()
```
> <img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/199910281-6a404591-7b2d-450b-913b-6f17372afd40.png?raw=true" width="320" height="256">

#### 7. NdviUiAdapter

The Normalized Difference Vegetation Index (NDVI) is an indicator of the presence of vegetation.
The index is commonly defined as：
<img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/200263841-3c8fb948-693f-40a0-925b-2cbbacbe3b7d.png?raw=true" width="100" height="50"> <br/>

where NIR is the reflectance in the near infrared (NIR) part of the spectrum and RED is the reflectance of the red band. For monarch, it takes band with CWL 713 nm for red band and band with CWL 805 nm for near infrared. In general, NDVI is an index to measure vegetation. When the NDVI value is high, the vegetation will be healthier. When the NDVI value is low, there is little or no vegetation.

```python
from unispectral.false_color import NdviUiAdapter
import matplotlib.pyplot as plt

adapter = NdviUiAdapter(image_mode="rgba", cache_dir=".cache")
adapter.load_cube("cube_20220826_145339"))
output = adapter.exec()
print(adapter.min_,adapter.max_,adapter.init_max_)
output = adapter.exec(0.3, 1)
fig, ax = plt.subplots(1, 1)
ax.imshow(output[:, :, :-1])
ax.set_axis_off()
plt.show()
```

> <img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/199911345-eac4291c-134a-4887-86b6-04a85d8891da.png?raw=true" width="320" height="256">
