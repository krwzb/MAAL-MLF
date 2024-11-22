# MAAL-MLF
This repository contains the official code implementation for the paper Modality-Aligned Anchor Learning for Scene Graph Generation. The full codebase will be released later.

## Installation
Check [INSTALL.md](https://github.com/krwzb/MAAL/blob/main/INSTALL.md) for installation instructions.

## Dataset
Check [DATASET.md](https://github.com/krwzb/MAAL/blob/main/DATASET.md) for instructions of dataset preprocessing.

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
We provide the log for the model. 

| Model |	mR@20 |	mR@50 |	mR@100 |	Google Drive |
| :--: | :--: | :--: | :--: | :--: |
|MAAL-MLF (PredCls)	| 30.4 |	36.4 |	38.6 |	[Log Link](https://drive.google.com/file/d/1FDwhXsH2bo9RJW0DuLoPPEosQbx4hYc1/view?usp=drive_link "PredCls_log") |
|MAAL-MLF (SGCls) |	17.1 |	20.5 |	21.7 |	[Log Link](https://drive.google.com/file/d/14W8DsSzDDJaZmKgoGvAH2yZrds__4DqL/view?usp=drive_link "SGCls_log") |
|MAAL-MLF (SGDet) |	13.7 |	15.6 |	16.8 |	[Log Link](https://drive.google.com/file/d/1jCrQpX9L-F8eWKMxFdfsMcL0Lk29y3zE/view?usp=drive_link "SGDet_log") |

## Visualization Results
![image](https://github.com/krwzb/MAAL/assets/166114889/29c7c819-c459-4e23-b440-c56dd75b1060)

## Train
```python
CUDA_VISIBLE_DEVICES=1 python MAAL-MLF/tools/relation_train_net.py \
--config-file "MAAL-MLF/configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
MODEL.ROI_RELATION_HEAD.PREDICTOR MAAL-MLF \
DTYPE "float32" \
SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 1 \
SOLVER.MAX_ITER 60000 SOLVER.BASE_LR 1e-3 \
SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
SOLVER.STEPS "(28000, 48000)" SOLVER.VAL_PERIOD 30000 \
SOLVER.CHECKPOINT_PERIOD 60000 GLOVE_DIR /home/MAAL-MLF/datasets/vg/glove \
MODEL.PRETRAINED_DETECTOR_CKPT /home/MAAL-MLF/datasets/vg/detector_model/pretrained_faster_rcnn/model_final.pth \
OUTPUT_DIR PENET/checkpoints/MAAL-MLF_PreCls \
SOLVER.PRE_VAL False \
SOLVER.GRAD_NORM_CLIP 5.0
```
## Test
```python
CUDA_VISIBLE_DEVICES=1 python MAAL-MLF/tools/relation_test_net.py \
--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
MODEL.ROI_RELATION_HEAD.PREDICTOR MAAL-MLF \
TEST.IMS_PER_BATCH 1 \
DTYPE "float32" \
GLOVE_DIR /home/MAAL-MLF/datasets/vg/glove \
MODEL.PRETRAINED_DETECTOR_CKPT /home/MAAL-MLF/datasets/vg/detector_model/pretrained_faster_rcnn/model_final.pth \
MODEL.WEIGHT checkpoints/MAAL-MLF_PredCls/model_final.pth \
OUTPUT_DIR checkpoints/MAAL-MLF_PredCls \
TEST.ALLOW_LOAD_FROM_CACHE False
```
## Device
All our experiments are conducted on one NVIDIA GeForce RTX 3090, if you wanna run it on your own device, make sure to follow distributed training instructions in [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).

## Acknowledgement

The code is implemented based on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).

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
