import sys
import os
sys.path.append(os.getcwd())
from modelConverter import Converter

#convert onnx model into tensort rt
Converter.tensorRT.onnx_to_tensorrt('model.onnx', 'model.plan', batch_size=8,  cuda_idx=0)

#Converter.tensorRT.onnx_to_tensorrt('model.onnx', 'model_GPU0.plan', batch_size=8,  cuda_idx=0)
#Converter.tensorRT.onnx_to_tensorrt('model.onnx', 'model_GPU1.plan', batch_size=8,  cuda_idx=1)

