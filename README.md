# Object_Detection using data fusion

### requirements
python 3.10.9,
pytorch 1.13.1,
Cuda 11.6,
gcc 11.2.0

### train and test
For the training FasterRCNNTrain.py file should be run.
Also, test_categorized.py should be run for evaluation(testing) on the test dataset.

#### rgb to fusion
TransfuserModel.py, class Transfuser, def forward

#### changing fusion operators
TransfuserModel.py, class Encoder, def forward

#### num input feature is changing (3 to 18)
TransfuserModel.py, class Encoder, self.lidar_encoder
