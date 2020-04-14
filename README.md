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

An complete application which perform face detection from camera then classify the emotion. The facedetection is computed on Movidius NCS and classification is on Edge TPU.

### Testing Movidius NCS Setup

To confirm movidius NCS setup. Run fllowing test app.

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

```
cd ../
python3 main.py
```

[![demo video](http://img.youtube.com/vi/6V4uWtrVqx0/0.jpg)](http://www.youtube.com/watch?v=6V4uWtrVqx0 "demo video")

## Training

* T.B.D

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

