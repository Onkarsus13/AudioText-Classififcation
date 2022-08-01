# AudioText-Classififcation
Audio Aware Text Classification


This is the multimodal task in which we are doing multi output classififcation using the information of both audio and text

For text-encoder we used the bert
for Audio-encoder we used LSTM

with the both features considration we had get the single feature vector which is having infomation of audio as well as text

To run the project

firsly download the dataset from [here](https://drive.google.com/file/d/1slGtHKHYTtiuC98yomV0hP3C85Q5V8sg/view)
Unzip the dataset

update the paths in `config.py`

then run: `pip install -r requirements.txt` to install all dependences

to train the model

`python train.py`

to get the f1 score on test set first download the model from [here](https://drive.google.com/file/d/1O2x97oRxDSzd0PgbY0BHOFrZr3_PY_Wl/view?usp=sharing)

`python evaluate.py`

all the tensorboard training logs can be found here

