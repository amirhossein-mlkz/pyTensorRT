import engine as eng
from engine  import Inference
from tensorflow import keras
import tensorrt as trt 
import numpy as np
import os
import threading
import time

class threadingInference(threading.Thread):
   def __init__(self, inf):
      threading.Thread.__init__(self)
      self.inf = inf

   def set_args(self, args): 
        self.args = args
   def run(self):

      self.outs = self.inf.do_inference(*self.args)
      #print ("Exiting " + self.args[0])




#_______________________________________________________________________________________
serialized_plan_fp32 = "resnet50_0.plan"
HEIGHT = 224
WIDTH = 224
batch = 128

inf0 = Inference( serialized_plan_fp32, cuda_idx=0 )
inf0.allocate_buffers(batch, trt.float32)
th_inf0 = threadingInference( inf0 )
#_______________________________________________________________________________________
serialized_plan_fp32 = "resnet50_1.plan"
HEIGHT = 224
WIDTH = 224
batch = 64

inf1 = Inference( serialized_plan_fp32, cuda_idx=1 )
inf1.allocate_buffers(batch, trt.float32)
th_inf1 = threadingInference( inf1 )
#_______________________________________________________________________________________

imgs0 = np.random.rand(batch, 3, HEIGHT, WIDTH).astype(np.float32 ) 
imgs1 = np.random.rand(batch, 3, HEIGHT, WIDTH).astype(np.float32 ) 

th_inf0.set_args( (imgs0 ,batch,HEIGHT,WIDTH) )
th_inf1.set_args( (imgs1 ,batch,HEIGHT,WIDTH) )

# th_inf1.start()
# th_inf0.start()

# th_inf0.join()
# th_inf1.join()


#_______________________________________________________________________________________
model = keras.applications.resnet.ResNet50(include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=1000)
model.load_weights('model.h5')
#------------------------------------------------------------
for i in range(500):
    
    imgs0 = np.random.rand(batch, 3, HEIGHT, WIDTH).astype(np.float32 ) 
    imgs1 = np.random.rand(batch, 3, HEIGHT, WIDTH).astype(np.float32 ) 
    t = time.time()
    inf0.do_inference( imgs0, batch, 224,224)
    #-------------------------------
 
    t = time.time() - t
    
    print(int(1/t * batch), t)

