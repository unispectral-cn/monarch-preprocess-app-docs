#### 1. Download the SDK.
You can download the unispectral sdk [v1.0.0](https://github.com/Unispectral-SW/monarch-preprocess-app-docs/releases/download/unispectral_sdk_v1.0.0/unispectral-1.0.0-cp38-cp38-win_amd64.whl)

#### 2. Install the SDK.
It is suggested to use python 3.8.10 and install the SDK with the following command.

```python
pip install unispectral-1.0.0-cp38-cp38-win_amd64.whl
```

Or you can accelerate the installation of dependent packages with the following command.
```python
pip install unispectral-1.0.0-cp38-cp38-win_amd64.whl -i https://pypi.douban.com/simple
``` 

#### 3. Quick start
You can start experiencing the SDK with the following sample code. The code will get the spectrum profile of a spectral cube. 
```python
from unispectral.spectrum import SpectrumRefUiAdapter

ref_adapter = SpectrumRefUiAdapter()
ref_adapter.load_cube(r"path_to_cube/ref_fps_4_20220826_143719")
ref_adapter.exec("Rect", [616, 389, 150, 140]) # [616, 389, 150, 140] is the ROI of the cube: [x, y, w, h]
print(ref_adapter.mean_)
```
> [177.19628571 224.10333333 225.09485714 380.31704762 357.29128571
 380.4987619  569.39028571 518.60404762 376.89366667 230.70785714]

 You can also use matplotlib to show the curve.

```python
import matplotlib.pyplot as plt

figure=plt.figure()
plt.plot(ref_adapter.mean_)
plt.show()
```

> <img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/2022-11-07_144508.png?raw=true" width="320" height="256">
 
 #### 4. Document
For more sample code and reference, please visit the [document website](https://unispectral-sw.github.io/monarch-preprocess-app-docs/#/).