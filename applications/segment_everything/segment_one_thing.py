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
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.gxf import Entity
from holoscan.operators import FormatConverterOp, HolovizOp, InferenceOp, V4L2VideoCaptureOp
from holoscan.resources import UnboundedAllocator


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

        # Transpose
        tensor = cp.asarray(in_message.get("preprocessed")).get()
        # OBS: Numpy conversion and moveaxis is needed to avoid strange
        # strides issue when doing inference
        tensor = np.moveaxis(tensor, 2, 0)[None]
        tensor = cp.asarray(tensor)

        # Create output message
        out_message = Entity(context)
        out_message.add(hs.as_tensor(tensor), "preprocessed")
        op_output.emit(out_message, "out")


class PostprocessorOp(Operator):
    """Operator to post-process inference output:
    * Non-max suppression
    * Make boxes compatible with Holoviz

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
        return




class SegmentOneThingApp(Application):
    def __init__(self, data, source="v4l2"):
        """Initialize the body pose estimation application"""

        super().__init__()

        # set name
        self.name = "Body Pose Estimation App"
        self.source = source

        if data == "none":
            data = os.path.join(
                os.environ.get("HOLOSCAN_DATA_PATH", "../data"), "body_pose_estimation"
            )

        self.sample_data_path = data

    def compose(self):
        pool = UnboundedAllocator(self, name="pool")

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

        inference_args = self.kwargs("inference")
        inference_args["model_path_map"] = {
            "yolo_pose": os.path.join(self.sample_data_path, "yolov8l-pose.onnx")
        }

        inference = InferenceOp(
            self,
            name="inference",
            allocator=pool,
            **inference_args,
        )

        postprocessor_args = self.kwargs("postprocessor")

        postprocessor = PostprocessorOp(
            self,
            name="postprocessor",
            allocator=pool,
            **postprocessor_args,
        )

        holoviz = HolovizOp(self, allocator=pool, name="holoviz", **self.kwargs("holoviz"))

        self.add_flow(source, holoviz, {(source_output, "receivers")})
        self.add_flow(source, preprocessor)
        self.add_flow(preprocessor, format_input)
        self.add_flow(format_input, inference, {("", "receivers")})
        self.add_flow(inference, postprocessor, {("transmitter", "in")})
        self.add_flow(postprocessor, holoviz, {("out", "receivers")})


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
