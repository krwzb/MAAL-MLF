# MSPN
This repository contains the official code implementation for the paper Multi-Scale Prototype Network for Scene Graph Generation.

## Installation
Check [INSTALL.md](https://github.com/krwzb/MSPN/blob/main/INSTALL.md) for installation instructions.

## Dataset
Check [DATASET.md](https://github.com/krwzb/MSPN/blob/main/DATASET.md) for instructions of dataset preprocessing.

Organize all the files like this:

```python
datasets
  |-- vg
    |--detector_model
      |--pretrained_faster_rcnn
        |--model_final.pth       
    |--glove
      |--.... (glove files, will autoly download)
    |--VG_100K
      |--.... (images)
    |--VG-SGG-with-attri.h5 
    |--VG-SGG-dicts-with-attri.json
    |--image_data.json    
```
## The Trained Model Weights
We provide the weights for the model. Due to random seeds and machines, they are not completely consistent with those reported in the paper, but they are within the allowable error range.

| Model |	mR@20 |	mR@50 |	mR@100 |	Google Drive |
| :--: | :--: | :--: | :--: | :--: |
|PE-Net (PredCls)	| 30.4 |	36.4 |	38.6 |	[Log Link](https://drive.google.com/file/d/1FDwhXsH2bo9RJW0DuLoPPEosQbx4hYc1/view?usp=drive_link "PredCls_log") |
|PE-Net (SGCls) |	17.1 |	20.5 |	21.7 |	[Log Link](https://drive.google.com/file/d/14W8DsSzDDJaZmKgoGvAH2yZrds__4DqL/view?usp=drive_link "SGCls_log") |
|PE-Net (SGDet) |	13.7 |	15.6 |	16.8 |	[Log Link](https://drive.google.com/file/d/1jCrQpX9L-F8eWKMxFdfsMcL0Lk29y3zE/view?usp=drive_link "SGDet_log") |

## The Trained Model Weights
We provide the weights for the model. Due to random seeds and machines, they are not completely consistent with those reported in the paper, but they are within the allowable error range.

| Model |	mR@20 |	mR@50 |	mR@100 |	Google Drive |
| :--: | :--: | :--: | :--: | :--: |
|PE-Net (PredCls)	| 30.4 |	36.4 |	38.6 |	[Log Link](https://drive.google.com/file/d/1FDwhXsH2bo9RJW0DuLoPPEosQbx4hYc1/view?usp=drive_link "PredCls_log") |
|PE-Net (SGCls) |	17.1 |	20.5 |	21.7 |	[Log Link](https://drive.google.com/file/d/14W8DsSzDDJaZmKgoGvAH2yZrds__4DqL/view?usp=drive_link "SGCls_log") |
|PE-Net (SGDet) |	13.7 |	15.6 |	16.8 |	[Log Link](https://drive.google.com/file/d/1jCrQpX9L-F8eWKMxFdfsMcL0Lk29y3zE/view?usp=drive_link "SGDet_log") |

## Train
```python
CUDA_VISIBLE_DEVICES=1 python MSPN/tools/relation_train_net.py \
--config-file "MSPN/configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
MODEL.ROI_RELATION_HEAD.PREDICTOR MSPN \
DTYPE "float32" \
SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 1 \
SOLVER.MAX_ITER 60000 SOLVER.BASE_LR 1e-3 \
SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
SOLVER.STEPS "(28000, 48000)" SOLVER.VAL_PERIOD 30000 \
SOLVER.CHECKPOINT_PERIOD 60000 GLOVE_DIR /home/MSPN/datasets/vg/glove \
MODEL.PRETRAINED_DETECTOR_CKPT /home/MSPN/datasets/vg/detector_model/pretrained_faster_rcnn/model_final.pth \
OUTPUT_DIR PENET/checkpoints/MSPN_PreCls \
SOLVER.PRE_VAL False \
SOLVER.GRAD_NORM_CLIP 5.0
```
## Test
```python
CUDA_VISIBLE_DEVICES=1 python MSPN/tools/relation_test_net.py \
--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
MODEL.ROI_RELATION_HEAD.PREDICTOR MSPN \
TEST.IMS_PER_BATCH 1 \
DTYPE "float32" \
GLOVE_DIR /home/MSPN/datasets/vg/glove \
MODEL.PRETRAINED_DETECTOR_CKPT /home/MSPN/datasets/vg/detector_model/pretrained_faster_rcnn/model_final.pth \
MODEL.WEIGHT checkpoints/MSPN_PredCls/model_final.pth \
OUTPUT_DIR checkpoints/MSPN_PredCls \
TEST.ALLOW_LOAD_FROM_CACHE False
```
## Device
All our experiments are conducted on one NVIDIA GeForce RTX 3090, if you wanna run it on your own device, make sure to follow distributed training instructions in Scene-Graph-Benchmark.pytorch.

## Tips
We use the rel_nms operation provided by RU-Net and HL-Net in PredCls and SGCls to filter the predicted relation predicates, which encourages diverse prediction results.

## Acknowledgement

The code is implemented based on Scene-Graph-Benchmark.pytorch.

## Citation
```python
@inproceedings{
  title={},
  author={},
  booktitle={},
  pages={},
  year={}

}
```
