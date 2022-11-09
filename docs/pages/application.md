#### 1. Training
You can use the [UNS Model Generator V1.0.exe](https://github.com/Unispectral-SW/monarch-preprocess-app-docs/releases/download/unispectral_sdk_v1.0.0/UNS.Model.Generator.V1.0.exe) to create your own model.

You can download the [peanut_traning_data](https://github.com/Unispectral-SW/monarch-preprocess-app-docs/releases/download/unispectral_sdk_v1.0.0/SDK_dataset.zip) and give it a try.


##### 1.1 Install UNS Model Generator
<img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/200252849-dafbf833-96ca-4f43-8ed8-22fd13ceef12.png?raw=true" width="600" height="500"><br/>

<img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/200252861-a57200bd-17d4-43ad-9829-a6565fd754e2.png?raw=true" width="600" height="500"><br/>

<img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/200252868-d237a4fd-aa33-4f40-b013-0e313d50a584.png?raw=true" width="600" height="500"><br/>
<img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/200252876-f43db26e-968d-4c7e-a0c9-7b3a8ad2caea.png?raw=true" width="600" height="500"><br/>

##### 1.2 Prepare the training dataset
The training dataset should be prepared in advance. The reference cubes and target objects cubes should be in the same folder.
<img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/200252908-0c02f8c5-cc1b-48a3-afb4-b2e298568a6a.png?raw=true" width="600" height="350"><br/>

**NOTE:**
The prefix of white reference cube should be **"ref"**.
It is highly suggested to use a black background when preparing the dataset and performing the classification.

<img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/200252922-ee36f30a-5463-4366-824f-4bcda9bd566a.png?raw=true" width="600" height="500"><br/>

##### 1.3 Select the dataset folder
<img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/200252936-61cd9b11-5087-4b09-8617-a4dc41f156cc.png?raw=true" width="600" height="500"><br/>

##### 1.4 Create labels
The label “Background” will be created automatically, so you only need to create the labels of target objects.
<img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/200252947-50392023-be1d-4be2-a18f-3647c24fa3fc.png?raw=true" width="600" height="500"><br/>

When finish creating the labels, you can start labeling for the cubes.

<img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/200252960-5118b3a6-c5f2-43e6-9e7d-4f986b0cf716.png?raw=true" width="600" height="500"><br/>

The following labeling window will pop up after clicking the “start labeling” button. You can start labeling for every target object.

**ShortCuts:**

w: start selecting ROI

d: navigate to next cube

a: navigate to previous cube

ctrl+a: save

<img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/200252978-c54e174d-3472-480c-842a-2b83b2503e98.png?raw=true" width="600" height="500"><br/>

**NOTE:**
Don’t forget to label the background.

<img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/200253002-36dc5826-7273-456d-a888-a62a637b6f2a.png?raw=true" width="600" height="500"><br/>

For the reference cube, please input the “Reference” as the label.

<img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/200253027-72a29f4e-2083-44d4-80c3-2f97e293202a.png?raw=true" width="600" height="500"><br/>

After labeling for all the cubes, you can close the labeling window.

<img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/200253035-cba64f5f-09e0-443e-b5a9-aba5394894b7.png?raw=true" width="600" height="500"><br/>

##### 1.5 Process Threshold mask

In some cases, there are many target objects in the cube. It’s time-consuming to do the labeling one by one. The threshold mask can be used to do labeling in a convenient and efficient way.

<img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/200253048-7c00268f-b606-4774-a82e-549f379e0f43.png?raw=true" width="600" height="500"><br/>

You can select multiple target objects in one ROI in the labeling window.

<img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/200253058-77c0e3b0-174e-4f29-8aec-1244c3207146.png?raw=true" width="600" height="500"><br/>

A dialog will pop up when closing the labeling window. If you want to process threshold mask, you can click “Yes”.

<img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/200253075-34bdf336-771b-47ec-8f9c-ec45c63693d9.png?raw=true" width="600" height="500"><br/>

The threshold mask window will pop up if you click “Yes” in the above step.

<img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/200253083-66b7c3ca-9e3f-4956-949f-2745b5cb2073.png?raw=true" width="600" height="500"><br/>

<img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/200253097-107052ea-1303-4da3-8425-3b6deef25544.png?raw=true" width="600" height="500"><br/>

The target objects in the ROI can be segmented by changing the threshold to a proper value.

##### 1.5 Create model

You can change the name, version, and the type of reference of the model.

**NOTE:**

If you want to use the model in Monarch Android APP, the name of the model should be as same as the name in Application.

<img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/200253116-d158f569-9ae0-47a5-b8bd-a45d280bf4b0.png?raw=true" width="600" height="500"><br/>

There are two types of references: big reference and small reference.
Big reference will do pixel normalization, small reference will do mean normalization when normalize the training data.

<img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/200253127-93397b9c-edb3-4ef0-8a88-807c7cf95ad3.png?raw=true" width="600" height="500"><br/>

Then you can click the “create model” button to generate the model.

<img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/200253141-749a4444-6a69-48b1-a05d-663709f1a030.png?raw=true" width="600" height="500"><br/>

If the model is generated successfully, the output folder will be opened automatically.

<img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/200253149-495b6109-56b7-4054-a402-d7efe8087919.png?raw=true" width="600" height="350"><br/>


#### 2. Prediction
You can download the [peanut_test_data](https://github.com/Unispectral-SW/monarch-preprocess-app-docs/releases/download/unispectral_sdk_v1.0.0/SDK_dataset.zip) and give it a try.

```python
adapter = ApplicationUiAdapter(image_mode="rgb")
adapter.load_app(model_path)
adapter.set_ref("ref_20221019_092834", RectRoi(656, 457, 130, 130))
adapter.load_cube("cube_20221019_091559")

output_img = adapter.exec()
origin_img = load_cube(cube_path).data[:, :, 4]

fig, ax = plt.subplots(1, 2)
fig.set_size_inches((10, 5))
ax[0].imshow(origin_img)
for name, color in adapter.legend_:
    ax[1].plot(0, 0, "-", label=name, color=color)
ax[1].legend(
    loc=1, prop={"size": 6}, frameon=True, edgecolor="#000000", facecolor="#FFFFFF"
)
ax[1].imshow(output_img)

plt.show()

```

> <img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/200758740-df3f029e-f6ad-438e-a61a-2e51c89c5c78.png?raw=true" width="900" height="400">
