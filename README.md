# LiteHPE: Head Pose Estimation
![Vizualization](https://github.com/Shohruh72/SixDRepNet/blob/master/weights/Result.gif)

## Features

* **6D Rotation Matrix Representation**
* **High Performance.**
* **Easy Integration**
* **Customizability**
  
*** _The project is structured into four main files:_**

- `main.py`: The entry point of the project, orchestrating the training, evaluation, and prediction processes.
- `nets.py`: Contains the definitions of neural network models or architectures.
- `datasets.py`: Manages dataset handling, including loading, preprocessing, and augmentations.
- `util.py`: Provides utility functions for data manipulation, visualization, logging, and other support tasks.

## Performance Metrics

The model achieved the following Mean Absolute Error (MAE) metrics across different pose angles:

### Results

| Backbone     | Epochs | Pitch | Yaw | Roll | Params (M) | FLOPS (M) | Pretrained weights |
|:------------:|:------:|------:|----:|-----:|-----------:|----------:|-------------------:|
| MobileOne_s0 |  120   |  5.4  | 4.2 | 3.8  |     2.1    |    275   | [model](https://github.com/Shohruh72/LiteHPE/releases/download/v.1.0.0/best_s0.pt)|
| MobileOne_s1 |  120   |  5.2  | 4.0 | 3.7  |     4.8    |    825   | [model](https://github.com/Shohruh72/LiteHPE/releases/download/v.1.0.0/best_s1.pt)|
| MobileOne_s2 |  120   |  4.9  | 3.7 | 3.4  |     7.8    |   1299   | [model](https://github.com/Shohruh72/LiteHPE/releases/download/v.1.0.0/best_s2.pt)|
| MobileOne_s3 |  120   |  4.7  | 3.6 | 3.3  |     10.1    |  1896   | [model](https://github.com/Shohruh72/LiteHPE/releases/download/v.1.0.0/best_s3.pt)|
| MobileOne_s4 |  120   |  4.6  | 3.5 | 3.2  |     14.8    |  2978   | [model](https://github.com/Shohruh72/LiteHPE/releases/download/v.1.0.0/best_s4.pt)|

## Installation

1. Clone the repository
2. Create a Conda environment using the environment.yml file:

```bash 
conda env create -f environment.yml
```

3. Activate the Conda environment:

```bash
conda activate HPE
```

## Preparing the Dataset

1. Download the 300W-LP, AFLW2000 Datasets:
2. Download the dataset from the
   official [project page](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm).
3. Place the downloaded dataset into a directory named 'Datasets'.

        
## Training the Model

To initiate the training process, use the following command:
* Configure your dataset path in main.py for training
* Configure Model name (default A2) in main.py for training
* Run the below command for Single-GPU training
```bash
python main.py --train
```
* Run the below command for Multi-GPU training $ is number of GPUs 
```bash
bash main.sh $ --train
```

## Testing the Model
Configure your dataset path in main.py for testing
Run the below command:
```bash
bash main.sh $ --test
```
## Inference
* Configure your video path in main.py for visualizing the demo
* Run the below command:
```bash
python main.py --demo
```



