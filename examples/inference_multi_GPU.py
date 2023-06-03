import sys
import os
sys.path.append(os.getcwd())
import numpy as np
from EngineRT import EngineRT, threadingInference


engine0 = EngineRT("model_GPU0.plan", cuda_idx=0)
engine1 = EngineRT("model_GPU1.plan", cuda_idx=0) #cuda_idx SHOULD BE 1




engine0_thread = threadingInference( engine0 )
engine1_thread = threadingInference( engine1 )

#_______________________________________________________________________________________
#Generate random inputs
imgs0 = np.random.rand(8, 3, 300, 300).astype(np.float32 ) 
imgs1 = np.random.rand(8, 3, 300, 300).astype(np.float32 ) 

#set inputs for each engine
engine0_thread.set_inputs( [imgs0]  )
engine1_thread.set_inputs( [imgs1]  )

#run threads
engine0_thread.start()
engine1_thread.start()

engine0_thread.join()
engine1_thread.join()

#get results
output0 = engine0_thread.results
output1 = engine1_thread.results
