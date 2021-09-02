# mHealth part 1 Activity Recognition: Prediction model

This repository is for the code to implement the prediction model of the activity recognition application
Designed to be used in the [Android Application](https://github.com/Zoltahn/mhealth-AR-App)
requires:

- latest version of [python](https://www.python.org/downloads/)
- latest version of [tensorflow](https://www.tensorflow.org/)
- [tensorflow lite](https://www.tensorflow.org/lite/) for model conversion
- ideally a CUDA enabled GPU

## Environment configuration
With python installed, install the packages with the following commands:

```
pip isntall tensorflow
pip install scipy
pip install sklearn
```

## Algorithms to compare:
#### Convolutional Neural Network(CNN):
Zeng, M., Nguyen, L. T., Yu, B., Mengshoel, O. J., Zhu, J., Wu, Pang., Zhang, Joy. (2015). Convolutional Neural Networks for human activity recognition using mobile sensors. In *6th International Conference on Mobile Computing, Applications and Services* (pp. 197-205) http://doi.org/10.4108/icst.mobicase.2014.257786

#### Recurrent Neural Network(RNN):
Inoue, M., Inoue, S. & Nishida, T. (2018). Deep recurrent neural network for mobile human activity recognition with high throughput. *Artificial Life and Robotics*, *23*(1), 173–185. https://doi.org/10.1007/s10015-017-0422-x

Benefit of using a Deep Learning model is that feature extraction does not need to be manually performed on raw data, which may save processing requirements on the android device thus more efficient power usage.
Aim is to implement both if possible and compare performance in terms of accuracy and power consumption.

## Datasets
currently looking at 2 datasets to train/verify:
- [WISDM](https://www.cis.fordham.edu/wisdm/dataset.php) Actitracker & Activity Prediction datasets 
  - Activity Prediciton dataset taking in controlled lab environment, Actitracker data collected in real life scenarios.
  - activities for: Standing, sitting, walking,  jogging, 'stairs', lying down
  - Each raw signal is labelled
- [UCI ML HAR](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) dataset 
  - activities forL Standing, sitting, walking, lying down, waling upstairs, walking downstairs
  - Raw signals grouped & labelled into 2.56s windows with a 50% overlap, at 50Hz sampling rate; 128 signals per window
  - Recorded using a Samsung Galaxy SII
  - Dataset has already been pre-processed using acceleromter and gyroscope noise filters
  - Split into 70% and 30% training and test sets, respectively.
  - Includes Walking, Sitting, Standing, Laying, Walking Upstairs, Walking Downstairs
