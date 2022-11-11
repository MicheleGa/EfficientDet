# EfficientDet: Scalable and Efficient Object Detection

This is a project intended to show EfficientDet, from Google Research and Brain team, at work.
It is a family of detectors that aims to achieve state-of-the-art accuracy while using fewer parameters than 
other state-of-the-art object detectors

Paper: [https://arxiv.org/abs/1911.09070](https://arxiv.org/abs/1911.09070)

In particular, the code is a fine-tuning of the model into the Global Wheat Detection dataset from Kaggle.

# How To Use

Install first all the dependencies in the *requirements.txt* file.
Then the entry point is in *main.py*, where you can adjust hyperparamenters
and decide which model to use.
Since my GPU resources were not sufficient, I used Google Colab (free account), in which I deployed a jupyter notebook that made the training and code demo.

# Third Party Code

- Dataset: [https://www.kaggle.com/c/global-wheat-detection](https://www.kaggle.com/c/global-wheat-detection)

- Model's implementation in Pytorch is made by Ross Wightman: [https://github.com/rwightman/efficientdet-pytorch](https://github.com/rwightman/efficientdet-pytorch)

- Code adapted from different sources: 
> - notebooks from course by [Giuseppe Lisanti](https://www.unibo.it/sitoweb/giuseppe.lisanti/en) and [Lorenzo Stacchio](https://www.unibo.it/sitoweb/lorenzo.stacchio2/en)
> - [https://www.kaggle.com/code/sadilkhan786/global-wheat-detection-pytorch](https://www.kaggle.com/code/sadilkhan786/global-wheat-detection-pytorch)
> - [https://www.kaggle.com/code/shonenkov/training-efficientdet/notebook](https://www.kaggle.com/code/shonenkov/training-efficientdet/notebook))
> - [https://medium.com/data-science-at-microsoft/training-efficientdet-on-custom-data-with-pytorch-lightning-using-an-efficientnetv2-backbone-1cdf3bd7921f](https://medium.com/data-science-at-microsoft/training-efficientdet-on-custom-data-with-pytorch-lightning-using-an-efficientnetv2-backbone-1cdf3bd7921f)