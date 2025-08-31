3D-CNN for Parkinson's Disease Detection from Gait
This repository contains a PyTorch implementation of the 3D Convolutional Neural Network (3D-CNN) for detecting Parkinson's Disease (PD) from gait videos, as described in the paper:

"Parkinson's disease detection from human gait using a deep 3D-convolutional neural network on video" by A. B. G. L. Pinto, et al.
Link to Paper

The goal of this project is to leverage deep learning to create a non-invasive diagnostic aid for Parkinson's Disease by analyzing patterns in how a person walks.

Model Architecture
The core of this project is the PDGaitAnalysisSystem, a 3D-CNN model built with PyTorch. The model is designed to process sequences of video frames (clips) to learn spatio-temporal features characteristic of parkinsonian gait.

The architecture consists of:

A CNN backbone with five Conv3d layers that progressively increase in channel depth (from 16 to 256). Each convolutional layer is followed by a ReLU activation and a MaxPool3d layer to downsample the feature maps.

A fully connected (FC) head that takes the flattened output from the CNN backbone and passes it through three Linear layers with ReLU activations and Dropout for regularization.

The final output layer classifies the input video clip into one of two classes: Parkinsonian Gait (PG) or Healthy Control (HC).

Dataset
The model expects the data to be pre-processed into individual frames and organized in a specific directory structure. The ParkinsonRGBDataset class automatically infers labels based on the folder names.

Expected Directory Structure:
<base_dataset_dir>/
├── Subject_01_PG/
│   └── frames/
│       └── video_01/
│           ├── frame_0001.png
│           ├── frame_0002.png
│           └── ...
├── Subject_02_HC/
│   └── frames/
│       └── video_01/
│           ├── frame_0001.png
│           ├── frame_0002.png
│           └── ...
└── ...
The label is considered 1 (Parkinson's) if "PG" is in the subject's folder name. Otherwise, it is 0 (Healthy).

The code processes videos by sampling a fixed number of frames (clip_length) from each video_01, video_02, etc. folder.

