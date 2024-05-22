# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
from argparse import ArgumentParser

import cupy as cp
import holoscan as hs
import numpy as np
from copy import deepcopy
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.gxf import Entity
from holoscan.operators import FormatConverterOp, HolovizOp, InferenceOp, V4L2VideoCaptureOp
from holoscan.resources import UnboundedAllocator


class DecoderInputData:
    def __init__(self, image_embeddings=None, point_coords=None, point_labels=None, mask_input=None, has_mask_input=None, orig_im_size=None, dtype=np.float16):
        
        self.image_embeddings = image_embeddings
        self.point_coords = point_coords
        self.point_labels = point_labels
        self.mask_input = mask_input
        self.has_mask_input = has_mask_input
        self.orig_im_size = orig_im_size
        self.dtype = dtype

    def __repr__(self) -> str:
        return f"DecoderInputData(image_embeddings={self.image_embeddings}, point_coords={self.point_coords}, point_labels={self.point_labels}, mask_input={self.mask_input}, has_mask_input={self.has_mask_input}, orig_im_size={self.orig_im_size}), dtype={self.dtype})"

    @staticmethod
    def create_decoder_inputs_from(input_point=None, input_label=None, input_box=None, box_labels=None, dtype=np.float16):
        input_point = input_point
        input_label = input_label
        input_box = input_box
        box_labels = box_labels
        if input_point is None:
            input_point = np.array([[1000, 600]], dtype=dtype)
        if input_label is None:
            input_label = np.array([1], dtype=dtype)
        if input_box is None:
            input_box = np.array([800, 150, 1250, 800], dtype=dtype)
        if box_labels is None:
            box_labels = np.array([2, 3], dtype=dtype)

        onnx_box_coords = input_box.reshape(2, 2)
        onnx_box_labels = box_labels

        onnx_coord = np.concatenate([input_point, onnx_box_coords], axis=0)[None, :, :]
        onnx_label = np.concatenate([input_label, onnx_box_labels], axis=0)[None, :].astype(dtype)

        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=dtype)
        onnx_has_mask_input = np.zeros(1, dtype=dtype)

        return DecoderInputData(
            point_coords=onnx_coord,
            point_labels=onnx_label,
            mask_input=onnx_mask_input,
            has_mask_input=onnx_has_mask_input,
            dtype=dtype
        )

    @staticmethod
    def scale_coords(coords: np.ndarray, orig_height=1024, orig_width=1024, resized_height=1024, resized_width=1024, dtype=np.float32) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension
        """
        old_h, old_w = orig_height, orig_width
        new_h, new_w = resized_height, resized_width
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords.astype(dtype)


class DecoderConfigurator(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Output tensor names
        self.outputs = [
            "image_embeddings", "point_coords", "point_labels", "mask_input", "has_mask_input"
        ]
        self.decoder_input = DecoderInputData.create_decoder_inputs_from(dtype=np.float32)
        self.decoder_input.point_coords = DecoderInputData.scale_coords(self.decoder_input.point_coords, orig_height=1024, orig_width=1024, resized_height=1024, resized_width=1024,dtype=np.float32)
        self.decoder_input.orig_im_size = np.array([1024, 1024], dtype=np.float32)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("in")

        image_tensor = cp.asarray(in_message.get("image_embeddings")).get()
        # image_tensor = image_tensor.astype(cp.float32)
        # image_tensor = cp.ascontiguousarray(image_tensor)

        print(image_tensor.shape, image_tensor.dtype)
        print(self.decoder_input)
        # Get input message
        print(in_message)
        print(in_message.get("image_embeddings"))
        data = {
            "image_embeddings": image_tensor,
            "point_coords": self.decoder_input.point_coords,
            "point_labels": self.decoder_input.point_labels,
            "mask_input": self.decoder_input.mask_input,
            "has_mask_input": self.decoder_input.has_mask_input,
            # "orig_im_size": self.decoder_input.orig_im_size
        }
        # Create output message
        out_message = Entity(context)
        for output in self.outputs:
            out_message.add(hs.as_tensor(data[output]), output)
        op_output.emit(out_message, "out")

class FormatInferenceInputOp(Operator):
    """Operator to format input image for inference"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("in")
        print(in_message)
        print(in_message.get("preprocessed"))

        # Transpose
        tensor = cp.asarray(in_message.get("preprocessed")).get()
        print(tensor.shape)
        # # OBS: Numpy conversion and moveaxis is needed to avoid strange
        # # strides issue when doing inference
        tensor = np.moveaxis(tensor, 2, 0)[None]
        tensor = cp.asarray(tensor)
        print(tensor.shape)

        # Create output message
        out_message = Entity(context)
        out_message.add(hs.as_tensor(tensor), "encoder_tensor")
        op_output.emit(out_message, "out")

class Sink(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output,  context):
        in_message = op_input.receive("in")
        print("----------------------------SINK Start")
        print(in_message)
        try:
            print(type(in_message))
            if isinstance(in_message, dict):
                for key, value in in_message.items():
                    print(f"{key}: {type(value)}")
                    try:
                        tensor = in_message.get(f"{key}")
                        print(tensor.shape)
                        print(tensor.dtype)

                    except Exception as e:
                        print(f"Could not get key {e}")
        except Exception as e:
            print(f"Could not get type, exception {e}")

        print("---------------------------SINK END")



class PostprocessorOp(Operator):
    """Operator to post-process inference output:
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Output tensor names


    def setup(self, spec: OperatorSpec):
        """
        input: "in"    - Input tensors coming from output of inference model
        output: "out"  - The post-processed output after applying thresholding and non-max suppression.
                         Outputs are the boxes, keypoints, and segments.  See self.outputs for the list of outputs.
        params:
            iou_threshold:    Intersection over Union (IoU) threshold for non-max suppression (default: 0.5)
            score_threshold:  Score threshold for filtering out low scores (default: 0.5)
            image_dim:        Image dimensions for normalizing the boxes (default: None)

        Returns:
            None
        """
        spec.input("in")
        spec.output("out")
        spec.param("iou_threshold", 0.5)
        spec.param("score_threshold", 0.5)
        spec.param("image_dim", None)

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("in")
        print(in_message)
        # Convert input to cupy array
        results = cp.asarray(in_message.get("decoder_output"))
        print(type(results))
        print(results.shape)
        return




class SegmentOneThingApp(Application):
    def __init__(self, data, source="v4l2"):
        """Initialize the body pose estimation application"""

        super().__init__()

        # set name
        self.name = "Segment one thing App"
        self.source = source

        if data == "none":
            data = os.path.join(
                os.environ.get("HOLOSCAN_DATA_PATH", "../data"), "body_pose_estimation"
            )

        self.sample_data_path = data

    def compose(self):
        pool = UnboundedAllocator(self, name="pool")

        sink = Sink(self, name="sink")

        if self.source == "v4l2":
            source = V4L2VideoCaptureOp(
                self,
                name="v4l2_source",
                allocator=pool,
                **self.kwargs("v4l2_source"),
            )
            source_output = "signal"

        format_input = FormatInferenceInputOp(
            self,
            name="transpose",
            pool=pool,
        )

        preprocessor_args = self.kwargs("preprocessor")
        preprocessor = FormatConverterOp(
            self,
            name="preprocessor",
            pool=pool,
            **preprocessor_args,
        )

        inference_encoder_args = self.kwargs("inference")
        inference_encoder_args["model_path_map"] = {
            "encoder": os.path.join("applications", "segment_everything", "engine_fp16", "encoder.engine"), 
        }

        inference = InferenceOp(
            self,
            name="inference",
            allocator=pool,
            **inference_encoder_args,
        )

        decoder_configurator = DecoderConfigurator(
            self,
            allocator=pool
            )


        inference_decoder_args = self.kwargs("inference_decoder")
        inference_decoder_args["model_path_map"] = {
            "decoder": os.path.join("applications", "segment_everything", "engine_fp16", "decoder.engine"), 
        }
        inference_decoder = InferenceOp(
            self,
            name="inference_decoder",
            allocator=pool,
            **inference_decoder_args,
        )

        postprocessor_args = self.kwargs("postprocessor")

        postprocessor = PostprocessorOp(
            self,
            name="postprocessor",
            allocator=pool,
            **postprocessor_args,
        )

        holoviz = HolovizOp(self, allocator=pool, name="holoviz", **self.kwargs("holoviz"))

        # # all connected
        # self.add_flow(source, holoviz, {(source_output, "receivers")})
        # self.add_flow(source, preprocessor)
        # # self.add_flow(preprocessor, holoviz), {("", "receivers")}
        # self.add_flow(preprocessor, format_input)
        # # self.add_flow(format_input, holoviz, {("out", "receivers")})
        # self.add_flow(format_input, inference, {("", "receivers")})
        # # self.add_flow(preprocessor, inference, {("", "receivers")})
        # # self.add_flow(inference, postprocessor), {"transmitter", "in"}
        # self.add_flow(inference, decoder_configurator, {("transmitter", "in")})
        # self.add_flow(decoder_configurator, inference_decoder, {("out", "receivers")})
        # # self.add_flow(decoder_configurator, postprocessor, {("out", "in")})
        # self.add_flow(inference_decoder, postprocessor, {("transmitter", "in")})
        # self.add_flow(postprocessor, holoviz, {("out", "receivers")})

        # # image flow only
        # self.add_flow(source, holoviz, {(source_output, "receivers")})

        # image flow and preprocessor
        # self.add_flow(source, holoviz, {(source_output, "receivers")})
        # self.add_flow(source, preprocessor)
        # self.add_flow(preprocessor, format_input)
        # self.add_flow(format_input, sink)

        # image to encoder
        # self.add_flow(source, holoviz, {(source_output, "receivers")})
        # self.add_flow(source, preprocessor)
        # self.add_flow(preprocessor, format_input)
        # self.add_flow(format_input, inference, {("", "receivers")})
        # self.add_flow(inference, sink)

        # up to decoder configurator
        self.add_flow(source, holoviz, {(source_output, "receivers")})
        self.add_flow(source, preprocessor)
        self.add_flow(preprocessor, format_input)
        self.add_flow(format_input, inference, {("", "receivers")})
        self.add_flow(inference, decoder_configurator, {("transmitter", "in")})
        self.add_flow(decoder_configurator, sink)

        # decoder too
        # self.add_flow(source, holoviz, {(source_output, "receivers")})
        # self.add_flow(source, preprocessor)
        # self.add_flow(preprocessor, format_input)
        # self.add_flow(format_input, inference, {("", "receivers")})
        # self.add_flow(inference, decoder_configurator, {("transmitter", "in")})
        # self.add_flow(decoder_configurator, inference_decoder, {("out", "receivers")})
        # self.add_flow(inference_decoder, sink)



if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser(description="Segment one thing application")
    parser.add_argument(
        "-s",
        "--source",
        choices=["v4l2"],
        default="v4l2",
        help=(
            "If 'v4l2', uses the v4l2 device specified in the yaml file."
        ),
    )
    parser.add_argument(
        "-c",
        "--config",
        default="none",
        help=("Set config path to override the default config file location"),
    )
    parser.add_argument(
        "-d",
        "--data",
        default="none",
        help=("Set the data path"),
    )
    args = parser.parse_args()

    if args.config == "none":
        config_file = os.path.join(os.path.dirname(__file__), "segment_one_thing.yaml")
    else:
        config_file = args.config

    app = SegmentOneThingApp(args.data, args.source)
    app.config(config_file)
    app.run()
