INPUT_SIZE=128
MODEL_NAME=face-reidentification-retail-0095-ft-with-mask

python -m onnxsim checkpoints/MobileFaceNetv2_mask/${MODEL_NAME}.onnx checkpoints/MobileFaceNetv2_mask/${MODEL_NAME}.onnx --input-shape 1,3,${INPUT_SIZE},${INPUT_SIZE}