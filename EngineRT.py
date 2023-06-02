# from tensorflow import keras
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import pycuda.autoinit 
import threading



LOG = True
TRT_DATE_TYPE = {
    'fp32' : trt.float32
}

class EngineRT:

   def __init__(self, engine_path,  cuda_idx=0, input_output_type='fp32'):
      
      #select GPU device by its index
      self.cuda_idx = cuda_idx
      self.cfx = cuda.Device( self.cuda_idx ).make_context()
      self.stream = cuda.Stream()
      if LOG:
         print('Selected Device is {}'.format(cuda.Device( self.cuda_idx ).name()))

      #--------------------------------------
      TRT_LOGGER = trt.Logger(trt.Logger.INFO)
      trt.init_libnvinfer_plugins(TRT_LOGGER, '' )
      self.trt_runtime = trt.Runtime(TRT_LOGGER)
      #--------------------------------------

      #load engine
      self.load_engine(engine_path)  

      self.context = self.engine.create_execution_context()

      # define data type of model for its input and output
      self.data_type = TRT_DATE_TYPE[input_output_type]

      # the first dimension of each input or output is batch size
      self.batch_size = self.engine.get_binding_shape(0)[0]

      #This function allocates the memory required for the inputs and outputs of the model
      self.allocate_buffers()
   


   def save(self, path:str):
      """save TensortRT model as .plan file

      Args:
          path (str): save path, it should be .plan file
      """
      assert '.plan' in path, "file's extention should be .plan"

      buf = self.engine.serialize()
      with open(path, 'wb') as f:
         f.write(buf)

       
   def load_engine(self, path:str):
      """load tensortRt model 

      Args:
          path (str): path of .plan file
      """
      assert '.plan' in path, "file's extention should be .plan"

      with open(path, 'rb') as f:
         engine_data = f.read()
      self.engine = self.trt_runtime.deserialize_cuda_engine(engine_data)
   #_________________________________________________________________________________________________________________________________________
   #
   #_________________________________________________________________________________________________________________________________________
   def allocate_buffers(self,):
      """This function is for allocating buffers for input and output in the GPU device
      """
      
      self.host_inputs = []
      self.cuda_inputs = []
      self.host_outputs = []
      self.cuda_outputs = []
      self.bindings = []
      #--------------------------
      self.inputs_shape = []
      self.outputs_shape = []

      #for loop on inputs and outputs of model
      for i in range(len(self.engine)):

         #get the input or output buffer shape. for e.g it may be like (8, 224, 244, 3) = (batch, height, width, channel) in resnet50
         buffer_shape = self.engine.get_binding_shape(i)
         
         #Calculation of the volume of input or output by multiplying its dimensions by each other  
         size = trt.volume( buffer_shape[:] ) * self.engine.max_batch_size

         # Allocate the required volume on host
         host_mem = cuda.pagelocked_empty(size, dtype=trt.nptype(self.data_type) )

         # Allocate the required volume on cuda
         cuda_mem = cuda.mem_alloc(host_mem.nbytes)

         self.bindings.append(int(cuda_mem))

         #check the buffer is input or output to append it into currect buffer list
         if self.engine.binding_is_input( self.engine[i] ):
            self.host_inputs.append(host_mem)
            self.cuda_inputs.append(cuda_mem)
            self.inputs_shape.append(buffer_shape)
         else:
            self.host_outputs.append(host_mem)
            self.cuda_outputs.append(cuda_mem)
            self.outputs_shape.append(buffer_shape)

      
   def load_images_to_buffer(self, inputs_data: np.ndarray, buffer):
      """copy data into buffer

      Args:
          inputs_data (np.ndarray): input data that you want copy into buffer
          pagelocked_buffer (_type_): the buffer that you want copy data to it
      """
      preprocessed = np.asarray(inputs_data).ravel()
      np.copyto(buffer, preprocessed) 

   #_________________________________________________________________________________________________________________________________________
   #
   #_________________________________________________________________________________________________________________________________________
   def inference(self, inputs_data:list[np.ndarray]) -> list[np.ndarray]:
      """ run model on inputs_data

      Args:
          inputs_data (list[np.ndarray]): list of inputs of model

      Returns:
          list[np.ndarray]: results of model
      """
      
      #threading.Thread.__init__(self)
      #self.cfx.push()

      #------------------------------------
      #load input data into host buffer
      for i in range(len(inputs_data)):
         self.load_images_to_buffer(inputs_data[i], self.host_inputs[i])
      #------------------------------------
      #copy host buffer into cuda buffer
      for i in range(len(self.host_inputs)):
         cuda.memcpy_htod_async( self.cuda_inputs[i], self.host_inputs[i], self.stream ) #send input host to cuda
      
      #execute model. the proccess would be done on hosts buffer 
      self.context.execute_async( bindings=self.bindings, stream_handle = self.stream.handle)

      for i in range(len(self.cuda_outputs)):
         cuda.memcpy_dtoh_async( self.host_outputs[i], self.cuda_outputs[i], self.stream)#get output cuda to host
      #------------------------------------
      self.stream.synchronize()
      #------------------------------------
      self.resault = []
      for i in range(len(self.host_outputs)):
         self.resault.append( 
            self.host_outputs[i].reshape((self.outputs_shape[i])) 
         )


      #self.cfx.pop()
      return self.resault

   
   #_________________________________________________________________________________________________________________________________________
   #
   #_________________________________________________________________________________________________________________________________________
   def destory(self):
      """destroy model and free GPU
      """
      self.cfx.pop()


class threadingInference(threading.Thread):
   def __init__(self, inf):
      threading.Thread.__init__(self)
      self.inf = inf

   def set_args(self, args): 
        self.args = args
   def run(self):

      self.outs = self.inf.do_inference(*self.args)
      #print ("Exiting " + self.args[0])


if __name__ == "__main__":
   engin = EngineRT('model.plan', cuda_idx=0)
   inpt = np.random.rand(8,300,300,3).astype(np.float32)
   engin.inference([inpt])
   x= 0