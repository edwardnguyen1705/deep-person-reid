"""
@Filename : fsanet
@Date : 2020-06-29
@Project: pymnn
@AUTHOR : NaviOcean
"""

from __future__ import print_function
import numpy as np
import MNN
import cv2
# from utils.box_utils import decode_np
# from utils.prior_box import PriorBox
# from utils.nms.py_cpu_nms import py_cpu_nms
import time
import os
import glob
import csv
import onnxruntime
from torchvision import datasets, transforms as T
from scipy.spatial import distance
from openvino.inference_engine import IENetwork, IEPlugin


norm_mean = [0., 0., 0.]
norm_std = [1., 1., 1.]

vec_lite = [[0.7656798, -0.20705204, 0.13403966, 0.18982321, 0.37042522, -0.032870412, -0.10552759, -0.04495398, 0.063819274, -0.400922, 0.13435033, 0.34666443, -0.045026835, 0.07916743, -0.16996035, -0.1675946, 0.25096896, -0.18640266, 0.23244709, -0.024957994, 0.09753327, -0.30105734, -0.05079846, 0.3727734, -0.31195965, 0.58773726, 0.4553581, -0.4012404, 0.02043807, -0.1349265, -0.090436615, 0.3109126, 0.17450908, 0.11300301, 0.15296598, -0.04769176, 0.24094786, -0.09604196, -0.39773336, -0.22267082, 0.04602332, -0.08568928, 0.006333339, 0.028039033, 0.4897525, 0.27429882, 0.17317085, 0.19691816, 0.4346613, -0.103715904, 0.02919324, 0.013929253, 0.06522305, -0.105788395, 0.11109193, 0.08748123, -0.38056508, 0.42565614, 0.31805798, 0.09283718, 0.34375098, 0.048153076, -0.124150194, 0.1491439, 0.43836024, 0.63074166, 0.23037456, -0.18854718, 0.14299317, 0.091047816, 0.32545388, -0.3003015, 0.08871619, 0.15036324, 0.09055104, -0.16629806, 0.20871283, -0.34487325, -0.36362094, -0.09783488, 0.081758924, 0.07915767, -0.16165614, -0.69130754, -0.35901713, 0.24611403, 0.21748844, 0.50862473, 0.15338968, 0.02335463, 0.05800738, -0.1536174, -0.38521683, 0.15625499, 0.3821005, 0.284571, -0.050479688, 0.13028795, 0.39383924, 0.18021548, 0.33112308, 0.2522309, 0.50167036, 0.1319843, -0.35602877, -0.036366638, 0.1951698, -0.14860304, 0.08341843, 0.2827403, -0.21511343, 0.29591295, 0.495056, 0.13174948, -0.17352372, 0.14609928, -0.021720469, 7.3215517E-4, 0.2655478, 0.08449443, 0.37036577, -0.41208932, -0.21529394, 0.33056277, -0.23653275, 0.116663516, -0.18040636, -0.014482559, -0.4368022, -0.47899333, 0.67763925, -0.5309965, -0.36523384, -0.14350002, 0.04374146, -0.67086446, 0.017199028, -0.012174973, 0.2848702, -0.13950866, -0.07276869, 0.11457928, 0.34785914, -0.0026744544, 0.153357, -0.06970247, -0.08577972, -0.042883977, -0.20858894, -0.6513742, -0.075904414, -0.12978065, -0.23794425, 0.2889139, -0.19577233, -0.19374633, -0.47299337, 0.47215915, 0.009725086, 0.18481675, 0.4508669, -0.34792376, 0.2848996, 0.173568, 0.044117883, -0.30306414, 0.25993496, 0.40282458, -0.048146002, -0.32617277, 0.09378984, 0.08145623, 0.20946057, 0.20514618, -0.089071214, 0.2355466, 0.08938206, 0.028200591, 0.031771712, -0.03271763, -0.2196125, 0.2792597, 0.31277815, -0.014550608, -0.0057443213, 0.027184261, 0.07019416, -0.7295973, -0.03614378, 0.14307068, -0.099760205, 0.019931976, 0.9049172, 0.11224929, -0.8202288, 0.117739625, -0.24344313, -0.1963078, -0.658339, 0.26569012, -0.14918588, -0.0021587992, 0.4062205, 0.026310956, 0.29875922, 0.26387998, 0.043912787, -0.16754043, 0.19365293, 0.14130169, 0.3528122, -0.22568387, -0.40161413, -0.39558655, 0.35682982, -0.24018046, -0.014624499, -0.18143786, 0.24657209, -0.04571539, -0.014476286, -0.015103902, 0.25695047, 0.3114731, 0.033967238, -0.021670569, 0.3932672, 0.22832744, 0.72019464, -0.08207561, 0.38438147, 0.28318697, 0.027824169, 0.003106759, -0.38353705, -0.10529014, -0.046628904, -0.41317183, -0.062271036, 0.00642199, -0.14722465, 0.015126318, 0.52409124, 0.6428334, 0.2594213, -0.31519893, -0.5121327, -0.063294195, -0.161986, -0.018896243, 0.0013681864, 0.4038577, -0.12396471, 0.2418624, 0.32427865, 0.17121762]]


def read_vec(vec_file):
    vec_lst = []
    with open(vec_file, 'r') as f:
        for line in f:
            vec_lst.append(float(line.rstrip()))
    return vec_lst


def getOutputData(output_tensor):
    output_shape = output_tensor.getShape()
    assert output_tensor.getDataType() == MNN.Halide_Type_Float
    tmp_output = MNN.Tensor(output_shape, output_tensor.getDataType(),
                            np.zeros(output_shape, dtype=float), output_tensor.getDimensionType())
    output_tensor.copyToHostTensor(tmp_output)
    output_data = np.array(tmp_output.getData(),
                           dtype=float).reshape(output_shape)

    return output_data


def preprocess(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = ((image / 255 - norm_mean) / norm_std).astype(np.float32)
    image = image.transpose(2, 0, 1)

    return image.astype(np.float32)


def preprocess_openvino(img_path):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (128, 128))
    image = image.transpose(2, 0, 1)

    return image.astype(np.float32)


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


def test_onnx():

    sess = onnxruntime.InferenceSession(
        "checkpoints/MobileFaceNetv2_mask/face-reidentification-retail-0095-ft-with-mask.onnx")
    ximg = preprocess('data/images/1181.jpg')
    n, c, h, w = sess.get_inputs()[0].shape
    ximg = ximg.reshape((1, 3, 128, 128))
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    result = sess.run(None, {input_name: ximg})
    # print("The model expects input shape: ", sess.get_inputs()[0].shape)
    # print("The shape of the Image is: ", ximg.shape)
    # print(result[0].transpose()[0][0][0])
    print(result[0])


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


def inference(vec_lite):
    """ inference mobilenet_v1 using a specific picture """
    interpreter = MNN.Interpreter("checkpoints/MobileFaceNetv2_mask/face-reidentification-retail-0095-ft-with-mask_fused.mnn")
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)
    # labels = read_csv()
    sub_folders = glob.glob('data/*')

    vec_file = 'mnn_cpp.txt'
    out_cpp = read_vec(vec_file)
    vec_lite = vec_lite[0]

    results = []
    for sub in sub_folders:
        imgs = glob.glob('{}/*.jpg'.format(sub))
        for img_path in imgs:
            print(img_path)
            img = preprocess(img_path)  # (3, 128, 128)

            tmp_input = MNN.Tensor((1, 3, 128, 128), MNN.Halide_Type_Float,
                                   img, MNN.Tensor_DimensionType_Caffe)
            # # construct tensor from np.ndarray
            input_tensor.copyFrom(tmp_input)
            interpreter.runSession(session)
            output_tensor = interpreter.getSessionOutput(
                session, "output").getData()
            # out = getOutputData(output_tensor)
            output = np.squeeze(output_tensor)
            # print(*output, sep=', ')
            dist = distance.cosine(output, np.asarray(vec_lite))
            print(dist)

            # out = out.transpose()[0][0].reshape(1, 256)[0]

            # label = labels[img_path]
            # result = [img_path, label]
            # result.extend(out)
            # results.append(result)
    # writeOutput(results)


def inference_openvino():
    model_xml = 'checkpoints/MobileFaceNetv2_mask/face-reidentification-retail-0095-ft-with-mask_fused.xml'
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    net = IENetwork(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.inputs))
    plugin = IEPlugin(device='CPU')
    exec_net = plugin.load(network=net)

    sub_folders = glob.glob('data/*')
    for sub in sub_folders:
        imgs = glob.glob('{}/*.jpg'.format(sub))
        for img_path in imgs:
            print(img_path)
            img = preprocess_openvino(img_path)  # (3, 128, 128)

            outputs = exec_net.infer(inputs={input_blob: img})
            key = list(outputs.keys())[0]
            output = outputs[key]
            output = np.squeeze(output)

            dist = distance.cosine(output, np.asarray(vec_lite))
            print(dist)


if __name__ == "__main__":
    inference_openvino()
    inference(vec_lite)
    # test_onnx()
    # read_csv()
    # vec_file = 'mnn_cpp.txt'
    # vec_lst = read_vec(vec_file)
    # print(vec_lst)
