# 🌋 AshenVenus 🌋

This repo is an open source entry to the [2023 Scroll Prize](https://scrollprize.org/) aka _The Vesuvius Challenge_. The name for this repo comes from [Venus](https://en.wikipedia.org/wiki/Venus_(mythology)#Iconography), who was the patron saint of [Pompeii](https://en.wikipedia.org/wiki/Pompeii), the city covered in Ash by the volcano Vesuvius.

![roman village at the foot of a large erupting volcano, ancient mosaic fresco, apocalypse, fantasy digital art, roman columns villa --v 5 --ar 2:1](assets/banner.png)

### Setup

Dependencies:

- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- HyperOpt
- PyTorch
- Tensorboard

```
pip install -r requirements.txt
```

### Training

To run a hyperparameter sweep use:

```
python sweep.py
```

To run a saved model use the evaluation notebook `eval.ipynb`. Follow the copy paste instructions to submit to Kaggle.

## YouTube

This repo was built live on YouTube, you can find the playlist here:

[![IMAGE_ALT](https://img.youtube.com/vi/J63V5n5OwMA/0.jpg)](https://youtube.com/playlist?list=PLwq2F0NejwX5Hc80-ExN9JfnbMAHR7HAn)

## Sources

Various sources used for reference:

- [PyTorch Pre-trained Models](https://pytorch.org/vision/main/models.html)
- [Pretrained ViT Multiscale Vision Transformers](https://arxiv.org/pdf/2104.11227.pdf)
- [Pretrained Video Transformer (SWIN)](https://github.com/pytorch/vision/blob/main/torchvision/models/video/swin_transformer.py)
- [A ConvNet for the 2020s](https://arxiv.org/pdf/2201.03545.pdf)
- [Adapting Pre-trained Vision Transformers from 2D to 3D through Weight Inflation Improves Medical Image Segmentation](https://proceedings.mlr.press/v193/zhang22a/zhang22a.pdf)
- [Segment Anything](https://github.com/facebookresearch/segment-anything)


## Citation

```
@misc{ashenvenus-vesuvius-challenge-2023,
  title={AshenVenus: Open Source Entry to the 2023 Scroll Prize},
  author={Hugo Ponte},
  year={2023},
  url={https://github.com/hu-po/ashenvenus}
}
```
