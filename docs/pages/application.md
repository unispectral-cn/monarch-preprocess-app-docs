#### 1. training
You can use the Unispectral SDK to generate your custom model file.
```python
from unispectral.datasets.parse import make_spcgroups_from_pascalvoc
from unispectral.datasets.parse import build_legend
from unispectral.datasets.spectral_cube import load_cube
from unispectral.preprocessing import Binning, Segmentation, Normalization
from unispectral.postprocessing import ConnectedCountDecision, ReversePool2d, RemoveEdge
from unispectral.modeling import PredictionPipeline, TrainingPipeline
from unispectral.application import save_app
from unispectral.application.adapter import ApplicationUiAdapter
from unispectral.datasets.geometry import RectRoi
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def training_model(voc_path, name, prefix, version, save_dir, norm_func):
    Xt = make_spcgroups_from_pascalvoc(voc_path)
    legend = build_legend(Xt)

    preprocess_pl = Pipeline(
        [
            (
                "segmentation_for_bg",
                Segmentation(
                    func=Segmentation.min_thresh_segmenter,
                    kw_args={"local": True, "kernel_size": kernel_size},
                ),
            ),
            (
                "binning",
                Binning(
                    func=Binning.mean_pool2d_remove_zero,
                    kw_args={"kernel_size": kernel_size},
                ),
            ),
            (
                "pix_norm",
                Normalization(
                    func=norm_func,
                ),
            ),
            (
                "pos_combine",
                Normalization(func=Normalization.pos_cube_combine),
            ),
        ],
    )

    modeling_pl = Pipeline(
        [
            (
                "rf",
                RandomForestClassifier(random_state=5),
            ),
        ]
    )

    postprocess_pl = Pipeline(
        [
            (
                "remove_edge",
                RemoveEdge(
                    func=RemoveEdge.remove_edge,
                    kw_args={"kernel_size_min_pool": 4},
                ),
            ),
            (
                "decision",
                ConnectedCountDecision(
                    func=ConnectedCountDecision.max_count_decision,
                    kw_args={"connect": True},
                ),
            ),
            (
                "reverse_pool",
                ReversePool2d(func=ReversePool2d.reverse_pool_2d),
            ),
        ]
    )


    # construct training pipeline and fit model
    training_pl = TrainingPipeline(preprocess_pl, modeling_pl)
    model = training_pl.fit_transform(Xt)

    # prediction pipeline
    prediction_pl = PredictionPipeline(preprocess_pl, model, postprocess_pl)

    # Save application
    save_app(
        save_dir=save_dir,
        name=name,
        version=version,
        legend=legend,
        prediction_pl=prediction_pl,
        prefix=prefix,
    )
```
#### 2. prediction

```python
adapter = ApplicationUiAdapter(image_mode="rgb")
adapter.load_app(model_path)
adapter.set_ref(ref_path, RectRoi(656, 457, 130, 130))
adapter.load_cube(cube_path)

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

# show accuracy
if adapter.label_xys:
    for label_xy, y_pred_cnt_pct in adapter.label_xys:
        ax[1].annotate(
            text="{:0.2f}".format(y_pred_cnt_pct),
            xy=label_xy,
            xytext=(-15, -2),
            textcoords="offset points",
            size=10,
            color="r",
        )
plt.show()

```

> <img src="https://github.com/Unispectral-SW/monarch-preprocess-app-docs/blob/main/docs/images/199926988-bdb0650f-7bff-4a7d-8099-47b2cecb4719.png" width="900" height="400">