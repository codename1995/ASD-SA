# Predicting atypical visual saliency for autism spectrum disorder via scale-adaptive inception module and discriminative region enhancement loss

This repository contains Keras implementation of our atypical visual saliency prediction model.

## Cite
Please cite with the following Bibtex code:
```
@article{wei2020predicting,
  title={Predicting atypical visual saliency for autism spectrum disorder via scale-adaptive inception module and discriminative region enhancement loss},
  author={Wei, Weijie and Liu, Zhi and Huang, Lijin and Nebout, Alexis and Le Meur, Olivier and Zhang, Tianhong and Wang, Jijun and Xu, Lihua},
  journal={Neurocomputing},
  volume={453},
  pages={610--622},
  year={2021},
  publisher={Elsevier}
}
```


## Pretrained weight on [Saliency4ASD](https://saliency4asd.ls2n.fr/)
[Google Drive](https://drive.google.com/file/d/1lqcmbsBT9pVPGLW847IJcE8KN3JTNEdU/view?usp=sharing)


## Training
Train model from scratch
```bash
$ python train.py --train_set_path path/to/training/set --val_set_path path/to/validation/set 
```
For training model based on our pretrained weight, please download the weight file and put it into `weights/`.
```bash
$ python train.py --train_set_path path/to/training/set --val_set_path path/to/validation/set --model_path weights/weights_DRE_S4ASD--0.9714--1.0364.pkl --dreloss False
```
The dataset directory structure should be 
```
└── Set  
    ├── Images  
    │   ├── 1.png  
    │   └── ...
    ├── FixMaps  
    │   ├── 1.png  
    │   └── ...
    ├── FixPts
    │   ├── 1.mat  
    │   └── ...
(If use DRE loss ...)
    ├── FixMaps_TD
    │   ├── 1.png  
    │   └── ...
    ├── FixPts_TD
        ├── 1.mat  
        └── ...
```
**Note:** We convert the `*_f.png` files in  `Saliency4ASD\TrainingDataset\AdditionalData\ASD_FixPts\` to MAT file by following code:
```Matlab
% Matlab Code
im = imread('1_f.png');
save('1.mat', 'im');
```


## Testing
Clone this repository and download the pretrained weights.

Then just run the code using 
```bash
$ python test.py --model-path weights/weights_DRE_S4ASD--0.9714--1.0364.pkl --images-path images/ --results-path results/
```
This will generate saliency maps for all images in the images directory and save them in results directory

## Requirements:
cuda 9.0  
cudnn 7.0  
python	3.5  
keras	2.2.2  
tensorflow	1.2.1  
opencv3	3.1.0  
matplotlib	2.0.2  

The detailed environment dependencies is in environment.yaml. You can easily copy the conda environment via
`conda env create -f environment.yaml`

## Comparison
It is recommended to compare with our model by online benchmarks, such as
* [MIT300](https://saliency.tuebingen.ai/results.html)
* [SALICON/LSUN'17](https://competitions.codalab.org/competitions/17136#results)
* [Saliency4ASD benchmark](https://saliency4asd.ls2n.fr/)

But if you are interested in the comparison with our model on the *Saliency4ASD 30*, you can refer to the `./DatasetPartition.txt` for the specific index of images. 

## Update:
### 2020/12/17
The original Saliency4ASD only contains FixPts in PNG format. We provide a simple code to convert the PNG file to MAT file for easy-using of our model.
### 2020/12/25
The test.py miss a line to sort the `file_name`. It has been fixed now.
### 2021/03/07
Add the index of images in training set, validation set and testing set in the ablation study.

## Acknowledgement
The code is heavily inspired by the following project:
1. SAM : https://github.com/marcellacornia/sam
2. EML-Net : https://github.com/SenJia/EML-NET-Saliency

Thanks for their contributions.

Many thanks to [@Imposingapple](https://github.com/Imposingapple) for pointing out a bug and fixing it.

## Contact 
If you have any questions, please contact me at codename1995@shu.edu.cn or my supervisor Prof. Zhi Liu at liuzhi@staff.shu.edu.cn.

## License 
This code is distributed under MIT LICENSE.
