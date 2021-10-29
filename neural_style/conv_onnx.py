import argparse
import os
import sys
import time
import re

import numpy as np
import torch.onnx

import coremltools as ct
from coremltools.models.neural_network import quantization_utils
import coremltools.proto.FeatureTypes_pb2 as ft

if __name__ == "__main__":
    styles = np.array(["candy", "mosaic", "rain_princess", "udnie"])
    for style in styles:
        model  = ct.converters.onnx.convert(model= style+'.onnx')
        model.save(style + ".mlmodel")
            # style_model.eval()

        # print(content_image.shape)
        # # print(content_image)

        # traced_model = torch.jit.trace(style_model, content_image)
        # model = ct.convert(traced_model, inputs=[ct.ImageType(name="image", shape=content_image.shape)])

        # spec = model.get_spec()
        # output = spec.description.output[0]

        # output.type.imageType.colorSpace = ft.ImageFeatureType.RGB
        # output.type.imageType.height = content_image.shape[2]
        # output.type.imageType.width = content_image.shape[2]

        # spec.save(style + ".mlmodel")

        # output = style_model(content_image).cpu()
        # utils.save_image(style + "_out.jpg", output[0])
