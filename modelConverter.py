import os
import tensorflow.keras.backend as K
import os
import tensorrt as trt
import pycuda.driver as cuda
from onnx import ModelProto
import tensorrt as trt 
import pycuda.autoinit #dont comment
import math




class onnxConverter:
    """Convert other model types into .onnx
        THIS VERSION ONLY SUPPORT .pb to .onnx
    """

    @staticmethod
    def pb_to_onnx(input_path:str, res_path:str) -> None:
        """convert .pb model into onnx 

        Args:
            input_path (str): path of .pb model folder ( Folder path )
            res_path (str): res path for save .onnx model ( .onnx file path)
        """
        assert os.path.isdir(input_path), "input path dosen't exist or is not a directory"
        assert '.onnx' in res_path, "res_path should be an .onnx file path" 
        
        command = 'python -m tf2onnx.convert --saved-model {} --output {}'.format(
            input_path,
            res_path
        )
        os.system(command)


class pbConverter:
    """covert models into .pb format
    """
    @staticmethod
    def kerasmodel_to_pb(model, res_path:str):
        """save keras model in .pb format

        Args:
            model (_type_): keras model
            res_path (str): res path that .pb files save into it
        """
        K.set_learning_phase(0) #may be not is neccesury
        paths = os.path.split(res_path)
        for i in range(len(paths)):
            __path__ = '/'.join(paths[:i+1])
            if not os.path.exists(__path__):
                os.mkdir(__path__)
    
        
        model.save(res_path)

    @staticmethod
    def h5_to_pb(model, res_path:str):
        assert False, "not supported in this version"




class tensorRtConverter:
    """covert models into .plan format ( tensorRT )
    """
    @staticmethod
    def onnx_to_tensorrt( onnx_path:str, res_path:str, batch_size:int, precision:str = 'fp16',  cuda_idx:int = 0, max_memory_size:int = 256 ) :
        """convert .onnx model to .plan model ( TensorRT model extention is .plan )

        Args:
            onnx_path (str): path of .onnx file
            onnx_path (str): res path for save tensorRT .plan file
            batch_size (int): ideal batch size of tensorRT model
            precision (str, optional): precision of tensorRT model. Defaults to 'fp16'.
            cuda_idx (int, optional): index of GPU device. Defaults to 0.
            max_memory_size (int, optional): maximum amount of memory that the model is allowed to cach in MB. it should be power of 2. Defaults to 256 MB.

        Returns:
            Engin: Engin Object
        """
        
        assert math.log(max_memory_size, 2).is_integer(), "max_memory_size should be power of 2"
        assert '.plan' in res_path, "file's extention in res_path should be .plan"

        print('Build Engine on {}'.format(cuda.Device( cuda_idx ).name()))

        #By executing these below two lines of code, the desired GPU will be activated and the engine will be built on it
        cfx = cuda.Device( cuda_idx ).make_context()
        stream = cuda.Stream()

        #read onnx model file
        model = ModelProto()
        with open(onnx_path, "rb") as f:
            model.ParseFromString(f.read())

        #creat input shape of model, [batch_size, dim0, dim1, ...]
        input_shape = [batch_size]
        for dim_idx in range(1, len(model.graph.input[0].type.tensor_type.shape.dim)):
            dim = model.graph.input[0].type.tensor_type.shape.dim[dim_idx].dim_value
            input_shape.append(dim)
        
        #-----------------------
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        trt_runtime = trt.Runtime(TRT_LOGGER)
        #-----------------------

        #build engin
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network( 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH )) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser:
            #The maximum amount of memory that the model is allowed to use
            # x<<20 means x * (2**20). 2*20 is equal 1048576 that is 1Mb 
            config.max_workspace_size = (max_memory_size << 20) #256 MiB
            
            #select precision of model
            if precision == 'fp16':
                #check fp16 is available
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                else:
                    assert True, 'precision fp16 is not allowed'
            
            elif precision == 'fp32':
                config.set_flag(trt.BuilderFlag.FP32)

            #load .onnx model
            with open(onnx_path, 'rb') as model:
                parser.parse(model.read())
            

            for i in range(parser.num_errors):
                print(parser.get_error(i))

    
            #set input shape
            network.get_input(0).shape = input_shape
            
            #convert .onnx to .plan (tensorRT)
            engine = builder.build_engine(network, config)

            #save tensorRT file
            buf = engine.serialize()
            with open(res_path, 'wb') as f:
                f.write(buf)

        return engine
        



#-----------------------------------------------------------------------------------------------------

class Converter:
    onnx = onnxConverter
    pb = pbConverter
    tensorRT = tensorRtConverter





if __name__=='__main__':
    

    pass



