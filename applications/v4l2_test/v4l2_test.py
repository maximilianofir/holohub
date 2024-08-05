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
from argparse import ArgumentParser


from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import (
    FormatConverterOp,
    HolovizOp,
    V4L2VideoCaptureOp,
)
from holoscan.resources import UnboundedAllocator
import cupy as cp


class SinkOp(Operator):
    def __init__(self, *args,  input_name="", **kwargs):
        super().__init__(*args, **kwargs)
        self.input_name = input_name

    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("in")
        print(in_message)
        print(type(in_message))
        print(in_message.keys())
        print(f"Received message: {in_message.get(self.input_name)}")
        if in_message.get(self.input_name) is None:
            print("Received message is None")
        else:
            # print shape and type of the input signal
            # Convert input to cupy array
            results = cp.asarray(in_message.get(self.input_name))
            print(f"Shape: {results.shape}, Type: {results.dtype}")


class V4L2(Application):
    def __init__(self, source="v4l2", video_device="none"):
        """Initialize"""

        super().__init__()

        # set name
        self.name = "v4l2 test App"
        self.source = source


        self.video_device = video_device

    def compose(self):
        pool = UnboundedAllocator(self, name="pool")

        # Input data type of preprocessor
        in_dtype = "rgb888"

        if self.source == "v4l2":
            v4l2_args = self.kwargs("v4l2_source")
            if self.video_device != "none":
                v4l2_args["device"] = self.video_device
            source = V4L2VideoCaptureOp(
                self,
                name="v4l2_source",
                allocator=pool,
                **v4l2_args,
            )
            source_output = "signal"
            # v4l2 operator outputs RGBA8888
            in_dtype = "rgba8888"

        preprocessor_args = self.kwargs("preprocessor")
        preprocessor = FormatConverterOp(
            self,
            name="preprocessor",
            pool=pool,
            in_dtype=in_dtype,
            **preprocessor_args,
        )

        holoviz_args = self.kwargs("holoviz")
        holoviz = HolovizOp(self, allocator=pool, name="holoviz", **holoviz_args)

        holoviz_args_two = self.kwargs("holoviz_two")
        holoviz_two = HolovizOp(self, allocator=pool, name="holoviz_two", **holoviz_args_two)

        sink = SinkOp(self, name="sink", allocator=pool, input_name="preprocessed")
        sink_source = SinkOp(self, name="sink_source", allocator=pool, input_name="source_video")

        # V4L2VideoCaptureOp → Holoviz = Ok
        self.add_flow(source, holoviz, {(source_output, "receivers")})
        
        # V4L2VideoCaptureOp → FormatConverter -->Holoviz = Ok
        self.add_flow(source, preprocessor, {(source_output, "source_video")})
        self.add_flow(preprocessor, holoviz_two, {("", "receivers")})

        # V4L2VideoCaptureOp → FormatConverter -->Sink = Ok
        self.add_flow(preprocessor, sink, {("", "in")})

        # V4L2VideoCaptureOp → sink = fail
        self.add_flow(source, sink_source, {(source_output, "in")})

        





if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser(description="Body Pose Estimation Application.")
    parser.add_argument(
        "-s",
        "--source",
        choices=["v4l2", "dds", "replayer"],
        default="v4l2",
        help=(
            "If 'v4l2', uses the v4l2 device specified in the yaml file or "
            " --video_device if specified. "
            "If 'dds', uses the DDS video stream configured in the yaml file. "
            "If 'replayer', uses video stream replayer."
        ),
    )
    parser.add_argument(
        "-c",
        "--config",
        default="none",
        help=("Set config path to override the default config file location"),
    )
    parser.add_argument(
        "-v",
        "--video_device",
        default="none",
        help=("The video device to use.  By default the application will use /dev/video0"),
    )

    args = parser.parse_args()

    if args.config == "none":
        config_file = os.path.join(os.path.dirname(__file__), "v4l2_test.yaml")
    else:
        config_file = args.config

    app = V4L2(args.source, args.video_device)
    app.config(config_file)
    app.run()
