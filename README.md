Overview

TransfuseNet introduces an innovative fusion technique combining LiDAR and camera data to improve 2D object detection accuracy. This approach diverges from traditional methods by emphasizing self-attention within Transformers to efficiently process and integrate multimodal inputs. Our novel Multi-Convolutional Fusion (MCF) strategy, coupled with a Priority Gate, optimizes feature selection during fusion stages, leading to superior detection performance.

Features

- Transformer-based architecture for efficient multimodal data integration.
- Novel Multi-Convolutional Fusion (MCF) strategy for enhanced feature extraction.
- Comprehensive evaluation on KITTI benchmark datasets, demonstrating competitive performance against state-of-the-art methods.

Requirements

- Python 3.6+
- PyTorch 1.7.1+
- torchvision 0.8.2+
- CUDA 10.1+ (For GPU acceleration)

### train and test
For the training FasterRCNNTrain.py file should be run.
Also, test_categorized.py should be run for evaluation(testing) on the test dataset.

#### rgb to fusion
TransfuserModel.py, class Transfuser, def forward

#### changing fusion operators
TransfuserModel.py, class Encoder, def forward

#### num input feature is changing (3 to 18)
TransfuserModel.py, class Encoder, self.lidar_encoder
