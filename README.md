# Edge Facial Sentiment Analysis

Facial sentiment analysis AI runs on a Raspberry Pi with Edge TPU.

## Getting Started

Here is a quick start guide to run with minimum effort using pretrained model.

```
# 1. Clone repository
$ git clone https://github.com/hamling-ling/FacialSentiment.git

# 2. Get pre-trained model
$ cd src/model
$ ./download.sh

# 3. Get example data
$ cd ../sample
$ ./download.sh

# 4. Run sentiment analysis
$ cd ../edge/test_edgetpu
$ python3 classify_image.py
```
You'll see output like this.
```
----INFERENCE TIME----
Note: The first inference on Edge TPU is slow because it includes loading the model into Edge TPU memory.
84.5ms
4 happiness: 0.99609
4.9ms
4 happiness: 0.99609
4.8ms
4 happiness: 0.99609
4.8ms
4 happiness: 0.99609
4.8ms
4 happiness: 0.99609
```

* above output is a classification result of this image

![happy.jpg](https://hailing-ling-public.s3-ap-northeast-1.amazonaws.com/GitHub/fascialsentiment/sample/happy.jpg "happy face")

### Prerequisites

#### Inference

- Raspberry Pi
- python3
- Edge TPU

#### Trainig

- Linux PC
- python3
- Tensorflow 1.15
- Jupyter Notebook
- PIL
- Image Magick

#### Application

- All of prerequistes for Inference
- Movidius NCS
- PiCamera

## Running Complete App

A complete application which performs face detection from camera then classifies the emotion into 9 categories. The face detection is computed on Movidius NCS and classification is on Edge TPU.

### Testing Movidius NCS Setup

To confirm movidius NCS setup, You can run fllowing test.

```
$ cd ../edge/test_movidius
$ python3 openvino_fd_myriad.py
```

Then you'll get an output like this.
```
[1.] [319  86  40  51]
[1.] [102 101  42  41]
[1.] [217  94  38  51]
```
And file out.png created looks like this

![out.png](https://hailing-ling-public.s3-ap-northeast-1.amazonaws.com/GitHub/fascialsentiment/output/out.png "face detection")

### Running The Main App

After everything is set, run following to start raspberry pi application.
```
cd ../
python3 main.py
```

[![demo video](http://img.youtube.com/vi/6V4uWtrVqx0/0.jpg)](http://www.youtube.com/watch?v=6V4uWtrVqx0 "demo video")

## Training

### Download Datasets

1. Go to following site and get fer2013.tar.gz. You need to register to Kaggle for free if you don't have an account.
   https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
2. I used labels from FER+. Grub fer2013new.csv from following link.
   https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
3. Create data/raw directory and plece following files.
   - data/raw/fer2013.csv
   - data/raw/fer2013new.csv

### Data Preparation
1. Open src/dataprep/create_datasets.ipynb in Jupyter notebook and run all cells.
   It creates roughly 30K files of 32x32 grayscale png images under data/input/ directory.
2. We need to convert those files into 96x96 8bit grayscale jpg.
```
$ cd src/dataprep
$ ./scale96x96grayjpg.sh
```
### Training
1. Open up src/train.ipynb in Jupyter notebook and run all. It takes more than 5 hours with GTX1060.
   Then you'll get following files.
   - model/labels.txt
   - model/builtin_mobilenetv2-longrun.h5
   - model/builtin_mobilenetv2-longrun.tflite
2. In the above training notebook, you will see the training curve like thks.
   ![facialsentiments30eps_up.png](https://hailing-ling-public.s3-ap-northeast-1.amazonaws.com/GitHub/fascialsentiment/output/facialsentiment30eps_up.png "training curve")
3. If you want to inference on PC, use h5 file and model/inference.ipynb
   And if you only want to convert from h5 to tflite, use convert.ipynb

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

