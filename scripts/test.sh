CUDA_VISIBLE_DEVICES=1 python MSPN/tools/relation_test_net.py \
--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
MODEL.ROI_RELATION_HEAD.PREDICTOR MAAL \
TEST.IMS_PER_BATCH 1 \
DTYPE "float32" \
GLOVE_DIR /home/MSPN/datasets/vg/glove \
MODEL.PRETRAINED_DETECTOR_CKPT /home/MAAL/datasets/vg/detector_model/pretrained_faster_rcnn/model_final.pth \
MODEL.WEIGHT checkpoints/MAAL_PredCls/model_final.pth \
OUTPUT_DIR checkpoints/MAAL_PredCls \
TEST.ALLOW_LOAD_FROM_CACHE False
