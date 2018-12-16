# volumenet
DEEP LEARNING FOR CLASSIFYING HEAVY DRINKERS FROM NORMAL CONTROLS with MRI Brain Volume Images

## Setup

`conda env create -f environment.yml`
conda install cython

# Source Code: 
loader.py : load dataset 
model.py : model definition
  NetFactory: create network 
  Net: base class
  AlexNet: Derived Class from Net
  SqueezeNet: Derived Class from Net
  VGG19 : Derived Class from Net
run_model.py
train.py
gradcam.py
misc_functions.py
evaluate.py

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




