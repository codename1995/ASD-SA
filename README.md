# Predicting atypical visual saliency for autism spectrum disorder via scale-adaptive inception module and discriminative region enhancement loss

This repository contains Keras implementation of our atypical visual saliency prediction model.

## Cite
TBC

<div style='display: none'>
Please cite with the following Bibtex code:
```
@inproceedings{ASD-SA2020,
  title={Predicting atypical visual saliency for autism spectrum disorder via scale-adaptive inception module and discriminative region enhancement loss},
  author={Wei, Weijie and Liu, Zhi and Huang, Lijin and Nebout, Alexis and Le Meur, Olivier and Zhang, Tianhong and Wang, Jijun and Xu, Lihua},
  booktitle={2019 IEEE International Conference on Multimedia \& Expo Workshops (ICMEW)},
  pages={621--624},
  year={2019},
  organization={IEEE}
}
```


## Pretrained weight on [Saliency4ASD](https://saliency4asd.ls2n.fr/)
[Google Drive](https://drive.google.com/file/d/1bK3CYLf_SVAmg1BMhgZgJ6fSDmQSgnkz/view?usp=sharing)

</div>

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

## Testing
Clone this repository and download the pretrained weights.

Then just run the code using 
```bash
$ python test.py --model-path weights/weights_DRE_S4ASD--0.9714--1.0364.pkl --images-path images/ --results-path results/
```
This will generate saliency maps for all images in the images directory and save them in results directory

## Requirements:
cuda 8.0  
cudnn 5.1  
python	3.5  
keras	2.2.2  
theano	0.9.0  
opencv	3.1.0  
matplotlib	2.0.2  

## Acknowledgement
The code is heavily inspired by the following project:
1. SAM : https://github.com/marcellacornia/sam
2. EML-Net : https://github.com/SenJia/EML-NET-Saliency

Thanks for their contributions.

## Contact 
If any question, please contact codename1995@shu.edu.cn

## License 
This code is distributed under MIT LICENSE.
