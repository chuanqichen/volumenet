# volumenet
DEEP LEARNING FOR CLASSIFYING HEAVY DRINKERS FROM NORMAL CONTROLS with MRI Brain Volume Images

## Setup

-conda env create -f environment.yml`
-conda install cython

# Source Code: 
- loader.py : load dataset 
- model.py : model definition
--   NetFactory: create network 
--  Net: base class
--  AlexNet: Derived Class from Net
--  SqueezeNet: Derived Class from Net
--  VGG19 : Derived Class from Net
- run_model.py
- train.py
- gradcam.py
- misc_functions.py
- evaluate.py

## Train
usage: train.py --rundir RUNDIR --model MODEL [--seed SEED] [--augment]
                [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY]
                [--epochs EPOCHS] [--max_patience MAX_PATIENCE]
                [--factor FACTOR]
Hyper Parameters 
- model : SqueezeNet , AlexNet, or VGG19 are supported 

For example: 
python train.py --model SqueezeNet --learning_rate=1e-5 --epochs=50 --max_patience=5 --factor=0.5 --weight_decay=0.05 --rundir subfolder
- arguments saved at `[subfolder]/args.json`
- models saved at `[subfolder]/[val_loss]_[train_loss]_epoch[epoch_num]`

## Evaluate

usage: evaluate.py --model_path MODEL_PATH --split SPLIT [--augment]
                   --model MODEL

- model : SqueezeNet , AlexNet, or VGG19 are supported 

For example: 
` python evaluate.py --split test --model SqueezeNet  --model_path SqueezeNet_002/models/SqueezeNet_val0.1577_train0.1701_epoch28


##References
[1] A. Pfefferbaum, T. Rohlfing, K. M. Pohl, B. Lane, W. Chu, D. Kwon, et al. Adolescent
development of cortical and white matter structure in the ncanda sample: Role of sex, ethnicity,
puberty, and alcohol drinking. Cereb Cortex., 26(10):4101–4121, 2016. doi: 10.1093/cercor/
bhv205.

[2] Adolf Pfefferbaum, Dongjin Kwon, et al. Altered brain developmental trajectories in adolescents
after initiating drinking. American Journal of Psychiatry., 175(4):370–380, 2017.

[3] Miles N. Wernick, Yongyi Yang, Jovan G. Brankov, Grigori Yourganov, and Stephen C. Strother.
Machine learning in medical imaging. IEEE Signal Process Mag., 27(4):25–38, 2010.

[4] Dinggang Shen, Guorong Wu, and Heung-Il Suk. Deep learning in medical image analysis.
Annual Review of Biomedical Engineering, 19:221–248, 2017.

[5] Eli Gibson, Wenqi Li, Carole Sudre, Lucas Fidon, et al. Niftynet: a deep-learning platform for
medical imaging. Computer Methods and Programs in Biomedicine, 158:113–122, 2018.

[6] Sang Hyun Park, Yong Zhang, Dongjin Kwon, Qingyu Zhao, et al. Alcohol use effects on
adolescent brain development revealed by simultaneously removing confounding factors, identifying
morphometric patterns, and classifying individuals. Nature Scientific Report, 8(8297),
2018.

[7] Neuroimaging python library. http://nipy.org/nibabel/. Read / write access to some
common neuroimaging file formats.

[8] The impact of squeezenet. https://aspire.eecs.berkeley.edu/wiki/_media/eop/
2017/bichen_talk_slides.pdf.

[9] Ilya Sutskever Krizhevsky, Alex and Geoffrey E. Hinton. Imagenet classification with deep
convolutional neural networks. Advances in neural information processing systems, 2012.

[10] Forrest N. Iandola. Squeezenet: Alexnet-level accuracy with 50x fewer parameters and< 1mb
model size. arXiv, 2016.


