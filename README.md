# Semantic_Segmentation_Project

## Project Description

This project was build for segmenting different body parts of a mosquito for identifying various species and combat vector-borne diseases.

```Due to privacy reason no link to the data is uploaded here!```

Since each mosquitio has eight distinct body parts, the problem becomes multiclass segmentation problem.
In this project, I experimented with the [UNET](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) model with various Backbones as Encoders from various RESNET architectures, namely RESNSET 34, 50, 101, and 152. Finally, after all of the training, RESNET 152 outperforms every other encoder.

# Technology Stack

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white) ![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)

# How to Run and train Locally

I have tested this project on my Windows 11 machine with 16GB RAM and NVDIA RTX 3070 GPU.Juast follow the instruction given below step by step for your custom data:

1. git clone using SSH (git@github.com:jaybfn/Semantic_Segmentation_Project.git)
2. Before running any code, take a look at the requirements.txt file and make sure you have all the packages installed.
3. the file ```trainUNET_Backbone.ipynb``` is a jupyter-notebook for testing your data at every line of the code, if no error is found then go to step 4.
4. 
