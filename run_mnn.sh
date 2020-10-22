MODEL_NAME=face-reidentification-retail-0095-ft-with-mask

mnnconvert -f ONNX \
    --modelFile checkpoints/MobileFaceNetv2_mask/${MODEL_NAME}.onnx \
    --MNNModel checkpoints/MobileFaceNetv2_mask/${MODEL_NAME}.mnn \
    --fp16 FP16