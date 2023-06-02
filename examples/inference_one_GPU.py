import sys
import os
sys.path.append(os.getcwd())
import numpy as np
from EngineRT import EngineRT


#load tensorRT engine
engine = EngineRT("model.plan", cuda_idx=0)

test_inputs = np.random.rand(8,300,300,3)

#inference inputs into model
result = engine.inference([test_inputs])

print(f" this model has {len(result)} output")
print(f" output shape: {result[0].shape}")
