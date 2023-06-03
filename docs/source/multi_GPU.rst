multi GPU
===================
in this section, we want to explain how using tensorRT on multi-GPU devices

step 1: build engines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

like before, we should convert our model to ``.onnx`` format. we ignore this step in this section. you can see how to do it for the TensorFlow model in :doc:`getting_start` .
then we should convert ``onnx`` into tensorRT engine. The important difference here is that you have to create **a separate tensorRT engine for each GPU**.
we do this by pass index of GPU device into ``cuda_idx`` argument of ``Converter.tensorRT.onnx_to_tensorrt`` method. 
GPU index for the first GPU device is equal to 0 and increases one by one for the other GPU devices.


.. code-block:: python

   from modelConverter import Converter

   #convert onnx model into tensort rt
   #Export TensorRT for first GPU devices
   Converter.tensorRT.onnx_to_tensorrt('model.onnx', 'model_GPU0.plan', batch_size=8,  cuda_idx=0)
   #Export TensorRT for second GPU devices
   Converter.tensorRT.onnx_to_tensorrt('model.onnx', 'model_GPU1.plan', batch_size=8,  cuda_idx=1)

After executing this part of the code, a file named ``model.plan`` will be created in the project path, which is tensorRT engine


step 2: inference TensorRt
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

like before, for using tensorRT engine, we use ``engineRT`` class of ``engineRT`` module. for more details see :doc:`EngineRT_src`. 
load the engines that you built in the previous step

.. note::
   Note that the value of the ``cuda_idx`` argument must be equal to the GPU index on which the engine is built.


.. code-block:: python

   import numpy as np
   from EngineRT import EngineRT

   #load tensorRT engines
   engine0 = EngineRT("model_GPU0.plan", cuda_idx=0)
   engine1 = EngineRT("model_GPU1.plan", cuda_idx=1)


Now we give an arbitrary input to the models with the ``inference`` method. 


.. code-block:: python

   # Generate random input. Its shape must match the input shape of the model
   test_inputs0 = np.random.rand(8,300,300,3)
   test_inputs1 = np.random.rand(8,300,300,3)

   #inference inputs into model
   result0 = engine0.inference([test_inputs0])
   result1 = engine1.inference([test_inputs1])



as you see, we used both GPU devices. but in practice, we want to infer on multi-GPU in parallel. for doing this, see the next chapter


parallel on multi-GPU (threading)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
in this section, we want to process our data on multi-GPU in parallel. let's load the models that we built in Step 1. 
also import ``threadingInference`` class from ``EngineRT``

.. code-block:: python

   import numpy as np
   from EngineRT import EngineRT, threadingInference


   #load tensorRT engines
   engine0 = EngineRT("model_GPU0.plan", cuda_idx=0)
   engine1 = EngineRT("model_GPU1.plan", cuda_idx=1)


now we should instance ``threadingInference`` object for each engine/

.. code-block:: python

   engine0_thread = threadingInference( engine0 )
   engine1_thread = threadingInference( engine1 )

now we generate two random inputs and process them in parallel

.. code-block:: python

   #Generate random inputs
   imgs0 = np.random.rand(8, 3, 300, 300).astype(np.float32 ) 
   imgs1 = np.random.rand(8, 3, 300, 300).astype(np.float32 ) 

   #set inputs for each engine
   engine0_thread.set_inputs( [imgs0]  )
   engine1_thread.set_inputs( [imgs1]  )

   #run threads
   engine0_thread.start()
   engine1_thread.start()

   #wait for finishing process
   engine0_thread.join()
   engine1_thread.join()

   #get results
   output0 = engine0_thread.results
   output1 = engine1_thread.results