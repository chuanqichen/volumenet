DEEP LEARNING FOR CLASSIFYING HEAVY DRINKERS FROM NORMAL CONTROLS with MRI Brain Volume Images

## Setup

-conda env create -f environment.yml

-conda install cython

## Dataset: 
Special permission is required to get access to NCANDA data after agreement is signed and access is approved. 

National Consortium on Alcohol and NeuroDevelopment in Adolescence (NCANDA) magnetic
resonance imaging (MRI) data is consisted of MRI images of total 808 adolescents, in which 674
adolescents meeting no/low alcohol or drug use criteria and 134 adolescents exceeding criteria.
Some adolescents have up to 3 annual scans as a result of the follow-up scans of each subject.


## Source Code: 
- loader.py : load dataset 
- model.py : model definition
                NetFactory: create network
                
                Net: base class
                
                VGG19 : Derived Class from Net
                
                AlexNet: Derived Class from Net
                
                SqueezeNet: Derived Class from Net
- run_model.py
- train.py
- gradcam.py
- misc_functions.py
- evaluate.py

## Train
usage: python train.py --rundir RUNDIR --model MODEL [--seed SEED] [--augment]

                [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY]
                
                [--epochs EPOCHS] [--max_patience MAX_PATIENCE]
                
                [--factor FACTOR]
                
Hyper Parameters 

- model : SqueezeNet , AlexNet, or VGG19 are supported 

For example: 

              python train.py --model SqueezeNet --learning_rate=1e-5 --epochs=50 --max_patience=5 
              --factor=0.5 --weight_decay=0.05 --rundir subfolder

              arguments saved at `[subfolder]/args.json`

              models saved at `[subfolder]/[val_loss]_[train_loss]_epoch[epoch_num]`

## Evaluate

usage: python evaluate.py --model_path MODEL_PATH --split SPLIT [--augment]
                   --model MODEL

- model : SqueezeNet , AlexNet, or VGG19 are supported 

For example: 

               python evaluate.py --split test --model SqueezeNet  --model_path                
                   SqueezeNet_002/models/SqueezeNet_val0.1577_train0.1701_epoch28

## Save training and evaluation results over epoches and visualization 

            TensorboardX (a library for pytorch) was used to save the intermediate results 
            (scalar datas, figures, images, graphcs) every epoches 
            
            Tensorboard can be used to load above results (scalar datas, figures, images, 
            graphcs) to visualize and monitor the training as it goes


## References

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

[11] Bien N, Rajpurkar P, Ball RL, Irvin J, Park A, Jones E, et al. (2018) Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet. PLoS Med 15(11): e1002699. https://doi.org/10.1371/journal.pmed.1002699

[12] Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, et al. Scikit-learn: machine learning in Python. J Mach Learn Res. 2011;12:2825–30.

[13] Paszke A, Gross S, Chintala S, Chanan G, Yang E, DeVito Z, et al. Automatic differentiation in PyTorch. 31st Conference on Neural Information Processing Systems; 2017 Dec 4–9; Long Beach, CA, US.

[14] van Rossum G. Python 2.7.10 language reference. Wickford (UK): Samurai Media; 2015.

[15] TensorboardX: https://github.com/lanpa/tensorboardX.

[16] Tensorboard: https://www.tensorflow.org/guide/summaries_and_tensorboard

