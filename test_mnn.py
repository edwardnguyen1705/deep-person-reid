from __future__ import print_function
import numpy as np
import MNN
import cv2
from utils.box_utils import decode_np
from utils.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import time
import os
import glob
import csv
import onnxruntime
from torchvision import datasets, transforms as T
​
norm_mean = [0., 0., 0.]
norm_std = [1., 1., 1.]
​
​


def getOutputData(output_tensor):
    output_shape = output_tensor.getShape()
    assert output_tensor.getDataType() == MNN.Halide_Type_Float
    tmp_output = MNN.Tensor(output_shape, output_tensor.getDataType(),
                            np.zeros(output_shape, dtype=float), output_tensor.getDimensionType())
    output_tensor.copyToHostTensor(tmp_output)
    output_data = np.array(tmp_output.getData(),
                           dtype=float).reshape(output_shape)


​
return output_data
​
​
def preprocess(img_path):


​
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (128, 128))
image = ((image / 255 - norm_mean) / norm_std).astype(np.float32)
image = image.transpose(2, 0, 1)
return image
​
​


def writeOutput(results):
    ids = list(range(256))
    fields = ['images', 'label']
    fields.extend(ids)
    f = open('output.csv', 'w')
    with f:
        writer = csv.writer(f)
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        row = dict()
        for result in results:
            for i in range(len(fields)):
                print(i, fields[i])
                row[fields[i]] = result[i]
            # print(row)
            writer.writerow(row)


​
​
def test_onnx():


​
sess = onnxruntime.InferenceSession(
    "Mobilenet_se_focal_121000.onnx")
ximg = preprocess('vec_images/1/4.jpg')
n, c, h, w = sess.get_inputs()[0].shape
ximg = ximg.reshape((1, 3, 128, 128))
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
result = sess.run(None, {input_name: ximg})
# print("The model expects input shape: ", sess.get_inputs()[0].shape)
# print("The shape of the Image is: ", ximg.shape)
# print(result[0].transpose()[0][0][0])
print(result[0])
​
​


def read_csv():
    results = dict()
    with open('query_pose_selected.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            label = row[-1]
            path = row[0]
            path = path.replace(
                '/home/phaihoang/Downloads/test_vectorize/AWL-V_aligned/', '')
            results[path] = int(label)
        return results


​
​


def inference():
    """ inference mobilenet_v1 using a specific picture """
    interpreter = MNN.Interpreter("MobileFaceNetv2_ft_JINS.mnn")
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)
    labels = read_csv()
    sub_folders = glob.glob('data/images/*')
    results = []
    for sub in sub_folders:
        imgs = glob.glob('{}/*.jpg'.format(sub))
        for img_path in imgs:
            img = preprocess(img_path)
            print(img_path)
            tmp_input = MNN.Tensor((1, 3, 128, 128), MNN.Halide_Type_Float,
                                   img, MNN.Tensor_DimensionType_Caffe)
            # # construct tensor from np.ndarray
            input_tensor.copyFrom(tmp_input)
            interpreter.runSession(session)
            output_tensor = interpreter.getSessionOutput(
                session, "output")
            out = getOutputData(output_tensor)
            print(out)
            out = out.transpose()[0][0].reshape(1, 256)[0]


​
exit()
label = labels[img_path]
result = [img_path, label]
result.extend(out)
results.append(result)
# writeOutput(results)
​
​
if __name__ == "__main__":
    inference()
