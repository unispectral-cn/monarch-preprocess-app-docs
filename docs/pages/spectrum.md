The Unispectral SDK provide the following adapters for developers to analysis spectral cube in a simple and convenient way.
#### 1. SpectrumRefUiAdapter

```python
from unispectral.spectrum import SpectrumRefUiAdapter

ref_adapter = SpectrumRefUiAdapter(ref_save_path=r".cache/ref_save.txt")
ref_adapter.load_cube("ref_fps_4_20220826_143719")
)
ref_adapter.exec("Rect", [616, 389, 150, 140])

```


#### 2. SpectrumObjUiAdapter

```python
from unispectral.spectrum import SpectrumObjUiAdapter

obj_adapter = SpectrumObjUiAdapter(ref_mean=ref_adapter.mean_)
obj_adapter.load_cube("cube_20220826_145339")
)
obj_adapter.exec("Rect", [576, 358, 40, 30])

```

#### 3. Reference normalization
Use SpectrumRefUiAdapter and SpectrumObjUiAdapter to get the spectrum ref normed signature.

```python
from unispectral.spectrum import SpectrumRefUiAdapter, SpectrumObjUiAdapter
import matplotlib.pyplot as plt

ref_adapter = SpectrumRefUiAdapter(ref_save_path=r".cache/ref_save.txt")
ref_adapter.load_cube("ref_fps_4_20220826_143719")
)
ref_adapter.exec("Rect", [616, 389, 150, 140])
print(ref_adapter.mean_)

obj_adapter = SpectrumObjUiAdapter(ref_mean=ref_adapter.mean_)
obj_adapter.load_cube("cube_20220826_145339")
)
obj_adapter.exec("Rect", [576, 358, 40, 30])
print(obj_adapter.mean_)

fig, ax = plt.subplots(1, 1)
ax.plot(obj_adapter.mean_)
plt.show()
```

> [177.21638095 224.10814286 225.10028571 380.29647619 357.28066667
 380.4987619  569.36857143 518.62090476 376.95638095 230.74433333]

>[0.92567158 1.05070926 1.01867189 1.06260955 1.09689674 1.03808704
 1.05413997 1.13149283 1.20043288 1.21961825]

> <img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/199899940-c0b9dc76-80d5-41db-9b28-61c4be745a19.png" width="320" height="256">
