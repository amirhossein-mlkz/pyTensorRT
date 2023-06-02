import sys
import os
sys.path.append(os.getcwd())
from modelConverter import Converter

#convert onnx model into tensort rt
Converter.tensorRT.onnx_to_tensorrt('model.onnx', 'model.plan', batch_size=8)

