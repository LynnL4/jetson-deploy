import os
import time
import tensorrt as trt
import pycuda.autoinit 
import pycuda.driver as cuda
import numpy as np
import argparse
import cv2

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

stream = cuda.Stream()

def parse_args():
    """Parse args from command line."""
    parser = argparse.ArgumentParser(description='TensorRT YOLOv8 Detector')
    parser.add_argument('model', type=str, default='yolov8n.engine', help='onnx model path')
    parser.add_argument('input', type=str, default='test.jpg', help='input image path')
    
    return parser.parse_args()


def generate_engine(onnx_file_path, engine_file_path, fp16_mode=False, int8_mode=False, save_engine=True):
    
    args = parse_args()
    
    builder = trt.Builder(TRT_LOGGER)
    
    network = builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    parser = trt.OnnxParser(network, TRT_LOGGER)
    ret = parser.parse_from_file(onnx_file_path)
    if not ret:
        for error in range(parser.num_errors):
            print(parser.get_error(error))
    
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30
    
    serialzed_engine = builder.build_engine(network, config)
    
    with open(engine_file_path, 'wb') as f:
        f.write(serialzed_engine.serialize())

def do_inference(engine_file_path, input_image):
    
    print('Loading engine from file {}...'.format(engine_file_path))
    
    with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

    input_shape = engine.get_binding_shape(0)
    output_shape = engine.get_binding_shape(1)
    h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=trt.nptype(trt.float32))
    h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=trt.nptype(trt.float32))
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    
    input = cv2.resize(input_image, (input_shape[2], input_shape[3])).transpose((2, 0, 1)).ravel()
        
    np.copyto(h_input, input)
    
    start = time.time()
        
    cuda.memcpy_htod_async(d_input, h_input, stream)
        
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
        
    stream.synchronize()
    
    output = h_output.reshape(engine.get_binding_shape(1))
        
    print("Time taken for inference is {}".format(time.time()-start))
    

def main():
    
    args = parse_args()
    
    if args.model.split('.')[-1] == 'onnx':
        generate_engine(args.model, args.model.split('.')[0]+'.engine')
    else:
        image = cv2.imread(args.input)
        do_inference(args.model, image)
    
    return


if __name__ == '__main__':
    main()

