import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import onnx
import argparse


from trt_utils import build_engine
shapes=[{"name": "mel", "min": (1, 80, 4),  "opt": (1, 80,768),  "max": (1, 80,1664)}]
print("Building melgan engine ...")
onnx = 'melgan.onnx'
onnx_engine = build_engine(onnx, shapes=shapes)
if onnx_engine is not None:
    with open("melgan.engine", 'wb') as f:
        f.write(onnx_engine.serialize())
        print("success to build engine from")
else:
    print("Failed to build engine from", onnx)

