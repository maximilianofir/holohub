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

import sys
#sys.path.append("/workspace/volumes/forks/holohub/benchmarks/holoscan_flow_benchmarking")
#from benchmarked_application import BenchmarkedApplication

import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import cupy as cp
import cupyx.scipy.ndimage
import holoscan as hs
import numpy as np
from copy import deepcopy
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.gxf import Entity
from holoscan.operators import FormatConverterOp, HolovizOp, InferenceOp, V4L2VideoCaptureOp
from holoscan.resources import UnboundedAllocator


def save_cupy_tensor(tensor, folder_path, counter=0, word="", verbose=False):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, f"{word}_{counter}.npy")
    cp.save(file_path, tensor)
    if verbose:
        print(f"Saved tensor to {file_path} \n")
        print(f"tensor dtype is {tensor.dtype}")


class DecoderInputData:
    def __init__(
        self,
        image_embeddings=None,
        point_coords=None,
        point_labels=None,
        mask_input=None,
        has_mask_input=None,
        orig_im_size=None,
        dtype=np.float32,
    ):

        self.image_embeddings = image_embeddings
        self.point_coords = point_coords
        self.point_labels = point_labels
        self.mask_input = mask_input
        self.has_mask_input = has_mask_input
        self.orig_im_size = orig_im_size
        self.dtype = dtype

    def print_ndims(self):
        for key, value in self.__dict__.items():
            if value is not None:
                print(f"{key}: {value.ndim}")

    def __repr__(self) -> str:
        return f"DecoderInputData(image_embeddings={self.image_embeddings}, point_coords={self.point_coords}, point_labels={self.point_labels}, mask_input={self.mask_input}, has_mask_input={self.has_mask_input}, orig_im_size={self.orig_im_size}), dtype={self.dtype})"

    @staticmethod
    def point_coords(point=None):
        if point is None:
            point = (500, 500)
        input_point = np.array([point], dtype=np.float32)
        input_label = np.array([1], dtype=np.float32)
        zero_point = np.zeros((1, 2), dtype=np.float32)
        # zero_point = input_point
        negative_label = np.array([-1], dtype=np.float32)
        coord = np.concatenate((input_point, zero_point), axis=0)[None, :, :]
        label = np.concatenate((input_label, negative_label), axis=0)[None, :]
        return coord, label

    @staticmethod
    def create_decoder_inputs_from(
        input_point=None, input_label=None, input_box=None, box_labels=None, dtype=np.float32
    ):

        onnx_coord, onnx_label = DecoderInputData.point_coords(input_point)
        if input_box is not None:
            input_box = input_box.reshape(2, 2)
            onnx_coord = np.concatenate([onnx_coord, input_box], axis=0)[None, :, :]
            onnx_label = np.concatenate([onnx_label, box_labels], axis=0)[None, :].astype(dtype)

        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=dtype)
        onnx_has_mask_input = np.zeros((1, 1, 1, 1), dtype=dtype)

        print(onnx_has_mask_input.ndim)

        return DecoderInputData(
            point_coords=onnx_coord,
            point_labels=onnx_label,
            mask_input=onnx_mask_input,
            has_mask_input=onnx_has_mask_input,
            dtype=dtype,
        )

    @staticmethod
    def scale_coords(
        coords: np.ndarray,
        orig_height=1024,
        orig_width=1024,
        resized_height=1024,
        resized_width=1024,
        dtype=np.float32,
    ) -> np.ndarray:
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
    def __init__(self, *args, save_intermediate=False, verbose=False, **kwargs):
        super().__init__(*args, **kwargs)
        if "input_point" in kwargs:
            input_point = kwargs["input_point"]
        else:
            input_point = None
        self.verbose = verbose
        self.save_intermediate = save_intermediate
        # Output tensor names
        self.outputs = [
            "image_embeddings",
            "point_coords",
            "point_labels",
            "mask_input",
            "has_mask_input",
        ]
        self.viz_outputs = ["point_coords"]
        self.decoder_input = DecoderInputData.create_decoder_inputs_from(
            dtype=np.float32, input_point=input_point
        )
        print(f"created inputs {self.decoder_input}")
        print(self.decoder_input)
        self.decoder_input.point_coords = DecoderInputData.scale_coords(
            self.decoder_input.point_coords,
            orig_height=1024,
            orig_width=1024,
            resized_height=1024,
            resized_width=1024,
            dtype=np.float32,
        )
        print(f"after scaling {self.decoder_input}")
        self.decoder_input.orig_im_size = np.array([1024, 1024], dtype=np.float32)
        print("---------------------init Decoder Config complete")

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")
        spec.output("point")

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("in")

        image_tensor = cp.asarray(in_message.get("image_embeddings"), order="C")
        if self.save_intermediate:
            # save the image embeddings
            save_cupy_tensor(
                folder_path="applications/segment_everything/downloads/numpy",
                tensor=image_tensor,
                word="image_embeddings",
                verbose=self.verbose,
            )

        if self.verbose:
            print(image_tensor.shape, image_tensor.dtype)
            print(self.decoder_input)
            # Get input message
            print(in_message)
            print(in_message.get("image_embeddings"))
            self.decoder_input.print_ndims()
        data = {
            "image_embeddings": image_tensor,
            "point_coords": cp.asarray(self.decoder_input.point_coords, order="C"),
            "point_labels": cp.asarray(self.decoder_input.point_labels, order="C"),
            "mask_input": cp.asarray(self.decoder_input.mask_input, order="C"),
            "has_mask_input": cp.asarray(self.decoder_input.has_mask_input, order="C"),
            # "orig_im_size": self.decoder_input.orig_im_sizep
        }
        # deep copy the point_coords
        copy_point_coords = cp.copy(data["point_coords"])
        # choose the first point
        copy_point_coords = copy_point_coords[0,0, :]
        # Create output message
        out_message = Entity(context)
        for i, output in enumerate(self.outputs):
            out_message.add(hs.as_tensor(data[output]), output)
        op_output.emit(out_message, "out")

        # create output for the point_coords
        point_message = Entity(context)
        point_message.add(hs.as_tensor(copy_point_coords), "point_coords")
        print(copy_point_coords)
        op_output.emit(point_message, "point")


class FormatInferenceInputOp(Operator):
    """Operator to format input image for inference"""

    def __init__(
        self, *args, mean=None, std=None, save_intermediate=False, verbose=False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self.mean = mean
        self.std = std
        if self.mean is None:
            self.mean = cp.array([123.675, 116.28, 103.53])
        if self.std is None:
            self.std = cp.array([58.395, 57.12, 57.375])
        self.save_intermediate = save_intermediate

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("in")
        if self.verbose:
            print("----------------------------------Inference Input")
            print(in_message)
            print(in_message.get("preprocessed"))

        # Transpose
        # Transpose
        tensor = cp.asarray(in_message.get("preprocessed"))
        # Normalize
        tensor = self.normalize_image(tensor)
        # reshape
        tensor = np.moveaxis(tensor, 2, 0)[np.newaxis]
        tensor = cp.asarray(tensor, order="C", dtype=cp.float32)
        tensor = cp.ascontiguousarray(tensor)

        # saving input
        if self.save_intermediate:
            save_cupy_tensor(
                folder_path="applications/segment_everything/downloads/numpy",
                tensor=tensor,
                word="input",
                verbose=self.verbose,
            )
        if self.verbose:
            print(f"---------------------------reformatted tensor shape: {tensor.shape}")

        # Create output message
        op_output.emit(dict(encoder_tensor=tensor), "out")

    def normalize_image(self, image):
        image = (image - self.mean) / self.std
        return image


class Sink(Operator):
    def __init__(self, *args, save_intermediate=False, verbose=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_intermediate = save_intermediate
        self.verbose = verbose

    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("in")
        if self.verbose:
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
                            # get a cupy tensor from the message and save it to a numpy array, using the key
                            if self.save_intermediate:
                                try:
                                    cupy_array = cp.asarray(tensor)
                                    save_cupy_tensor(
                                        cupy_array,
                                        folder_path="applications/segment_everything/downloads/numpy",
                                        counter=0,
                                        word=key,
                                        verbose=self.verbose,
                                    )
                                except Exception as e:
                                    print(f"Could not save cupy array {e}")

                        except Exception as e:
                            print(f"Could not get key {e}")
            except Exception as e:
                print(f"Could not get type, exception {e}")

            print("---------------------------SINK END")


class CupyArrayPainter:
    def __init__(self, colormap: cp.ndarray = None):
        if colormap is None:
            colormap = plt.get_cmap("viridis")
            colormap = colormap(np.linspace(0, 1, 256))
            colormap = cp.asarray(colormap * 255, dtype=cp.uint8)
        self.colormap = colormap

    def normalize_data(self, data):
        min_val = data.min()
        max_val = data.max()
        normalized_data = (data - min_val) / (max_val - min_val)
        return normalized_data

    def apply_colormap(self, data):
        # Scale normalized_data to index into the colormap
        indices = (data * (self.colormap.shape[0] - 1)).astype(cp.int32)

        # Get the RGB values from the colormap
        rgba_image = self.colormap[indices]
        return rgba_image

    def to_rgba(self, data):
        normalized_data = self.normalize_data(data)
        rgba_image = self.apply_colormap(normalized_data)
        return rgba_image


class PostprocessorOp(Operator):
    """Operator to post-process inference output:"""

    def __init__(self, *args, save_intermediate=False, verbose=False, **kwargs):
        super().__init__(*args, **kwargs)
        # Output tensor names
        self.outputs = ["out_tensor"]
        self.slice_dim = 3
        self.transpose_tuple = None
        self.threshold = None
        self.cast_to_uint8 = False
        self.counter = 0
        self.painter = CupyArrayPainter()
        self.save_intermediate = save_intermediate
        self.verbose = verbose

    def setup(self, spec: OperatorSpec):
        """
        input: "in"    - Input tensors coming from output of inference model
        output: "out"  -

        Returns:
            None
        """
        spec.input("in")
        spec.output("out")

    def mask_to_rgba(self, tensor, channel_dim=-1, color=None):
        """convert a tensor of shape (1, 1, 1024, 1024) to a tensor of shape (1, 3, 1024, 1024) by repeating the tensor along the channel dimension
        assuming the input tensor is a mask tensor, containing 0s and 1s.
        set a color for the mask, yellow by default.
        if color has length 3, it will be converted to a 4 channel tensor by adding 255 as the last channel
        the last number in the color tuple is the alpha channel

        Args:
            tensor (_type_): tensor with mask
            channel_dim (int, optional): dimension of the channels. Defaults to -1.
            color (tuple, optional): color for display. Defaults to (255, 255, 0).

        Returns:
            _type_: _description_
        """
        # check that the length of the color is 4
        if color is None:
            color = cp.array([255, 255, 0, 128], dtype=cp.uint8)
        assert len(color) == 4, "Color should be a tuple of length 4"
        tensor = cp.concatenate([tensor] * 4, axis=channel_dim)
        tensor[tensor == 1] = color
        return tensor

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("in")
        # Convert input to cupy array
        results = cp.asarray(in_message.get("low_res_masks"))
        if self.save_intermediate:
            save_cupy_tensor(
                folder_path="applications/segment_everything/downloads/numpy",
                tensor=results,
                counter=self.counter,
                word="low_res_masks",
                verbose=self.verbose,
            )
        if self.verbose:
            print(results.flags)
            print("-------------------postprocessing")
            print(type(results))
            print(results.shape)

        # scale the tensor
        scaled_tensor = self.scale_tensor_with_aspect_ratio(results, 1024)
        if self.verbose:
            print(f"Scaled tensor {scaled_tensor.shape}\n")
            print(scaled_tensor.flags)
        if self.save_intermediate:
            save_cupy_tensor(
                folder_path="applications/segment_everything/downloads/numpy",
                tensor=scaled_tensor,
                counter=self.counter,
                word="scaled",
                verbose=self.verbose,
            )

        # undo padding
        unpadded_tensor = self.undo_pad_on_tensor(scaled_tensor, (1024, 1024)).astype(cp.float32)
        if self.verbose:
            print(f"unpadded tensor {unpadded_tensor.shape}\n")
            print(unpadded_tensor.flags)

        if self.slice_dim is not None:
            unpadded_tensor = unpadded_tensor[:, self.slice_dim, :, :]
            unpadded_tensor = cp.expand_dims(unpadded_tensor, 1).astype(cp.float32)

            if self.save_intermediate:
                save_cupy_tensor(
                    folder_path="applications/segment_everything/downloads/numpy",
                    tensor=unpadded_tensor,
                    counter=self.counter,
                    word="sliced",
                    verbose=self.verbose,
                )

        if self.transpose_tuple is not None:
            unpadded_tensor = cp.transpose(unpadded_tensor, self.transpose_tuple).astype(cp.float32)

            if self.save_intermediate:
                save_cupy_tensor(
                    folder_path="applications/segment_everything/downloads/numpy",
                    tensor=unpadded_tensor,
                    counter=self.counter,
                    word="transposed",
                    verbose=self.verbose,
                )

        # threshold the tensor
        if self.threshold is not None:
            unpadded_tensor = cp.where(unpadded_tensor > self.threshold, 1, 0).astype(cp.float32)
            if self.save_intermediate:
                save_cupy_tensor(
                    folder_path="applications/segment_everything/downloads/numpy",
                    tensor=unpadded_tensor,
                    counter=self.counter,
                    word="thresholded",
                    verbose=self.verbose,
                )

        # cast to uint8 datatype
        if self.cast_to_uint8:
            print(unpadded_tensor.flags)
            unpadded_tensor = cp.asarray(unpadded_tensor, dtype=cp.uint8)
            if self.save_intermediate:
                save_cupy_tensor(
                    folder_path="applications/segment_everything/downloads/numpy",
                    tensor=unpadded_tensor,
                    counter=self.counter,
                    word="casted",
                    verbose=self.verbose,
                )

        if self.verbose:
            print(
                f"unpadded_tensor tensor, casted to {unpadded_tensor.dtype} and shape {unpadded_tensor.shape}\n"
            )

        # save the cupy tensor
        if self.save_intermediate:
            save_cupy_tensor(
                folder_path="applications/segment_everything/downloads/numpy",
                tensor=unpadded_tensor,
                counter=self.counter,
                word="unpadded",
                verbose=self.verbose,
            )

        # Create output message
        # create tensor with 3 dims for vis, by squeezing the tensor in the batch dimension
        unpadded_tensor = cp.squeeze(unpadded_tensor, axis=(0, 1))

        unpadded_tensor = self.painter.to_rgba(unpadded_tensor)

        # unpadded_tensor  = self.mask_to_rgba(unpadded_tensor)
        # make array ccontiguous
        unpadded_tensor = cp.ascontiguousarray(unpadded_tensor)

        out_message = Entity(context)
        for output in self.outputs:
            out_message.add(hs.as_tensor(unpadded_tensor), output)
        op_output.emit(out_message, "out")

    def scale_tensor_with_aspect_ratio(self, tensor, max_size, order=1):
        # assumes tensor dimension (batch, height, width)
        height, width = tensor.shape[-2:]
        aspect_ratio = width / height
        if width > height:
            new_width = max_size
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = max_size
            new_width = int(new_height * aspect_ratio)

        scale_factors = (new_height / height, new_width / width)
        # match the rank of the scale_factors to the tensor rank
        scale_factors = (1,) * (tensor.ndim - 2) + scale_factors
        # resize the tensor to the new shape using cupy
        scaled_tensor = cupyx.scipy.ndimage.zoom(tensor, scale_factors, order=order)

        # scaled_tensor = torch.nn.functional.interpolate(tensor, size=scaled_tensor_shape, mode='bilinear', align_corners=False)
        return scaled_tensor

    def undo_pad_on_tensor(self, tensor, original_shape):
        if isinstance(tensor, cp.ndarray):
            # get number of dimensions
            n_dims = tensor.ndim
        else:
            n_dims = tensor.dim()
        width, height = original_shape[:2]
        # unpad the tensor
        if n_dims == 4:
            unpadded_tensor = tensor[:, :, :height, :width]
        elif n_dims == 3:
            unpadded_tensor = tensor[:, :height, :width]
        else:
            raise ValueError("Invalid tensor dimension")
        return unpadded_tensor


class SegmentOneThingApp(Application):
    def __init__(self, data, source="v4l2", save_intermediate=False, verbose=False):
        """Initialize the body pose estimation application"""

        super().__init__()

        # set name
        self.name = "Segment one thing App"
        self.source = source
        self.verbose = verbose
        self.save_intermediate = save_intermediate
        if data == "none":
            data = os.path.join(
                os.environ.get("HOLOSCAN_DATA_PATH", "../data"), "body_pose_estimation"
            )

        self.sample_data_path = data

    def compose(self):
        pool = UnboundedAllocator(self, name="pool")

        sink = Sink(
            self, name="sink", save_intermediate=self.save_intermediate, verbose=self.verbose
        )

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
            verbose=self.verbose,
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
            "encoder": os.path.join(
                "applications", "segment_everything", "engine_fp32", "encoder.engine"
            ),
        }

        inference = InferenceOp(
            self,
            name="inference",
            allocator=pool,
            **inference_encoder_args,
        )
        # decoder_configurator_args = self.kwargs("decoder_configurator")
        decoder_configurator = DecoderConfigurator(
            self,
            allocator=pool,
            save_intermediate=self.save_intermediate,
            verbose=self.verbose,
            **self.kwargs("decoder_configurator"),
        )
        decoder_configurator.decoder_input.print_ndims()

        inference_decoder_args = self.kwargs("inference_decoder")
        inference_decoder_args["model_path_map"] = {
            "decoder": os.path.join(
                "applications", "segment_everything", "engine_fp32", "decoder.engine"
            ),
        }
        inference_decoder = InferenceOp(
            self,
            name="inference_decoder",
            allocator=pool,
            **inference_decoder_args,
        )

        postprocessor = PostprocessorOp(
            self,
            name="postprocessor",
            allocator=pool,
            save_intermediate=self.save_intermediate,
            verbose=self.verbose,
        )

        holoviz = HolovizOp(self, allocator=pool, name="holoviz", **self.kwargs("holoviz"))

        # Holoviz
        self.add_flow(source, holoviz, {(source_output, "receivers")})
        self.add_flow(source, preprocessor)
        self.add_flow(preprocessor, format_input)
        self.add_flow(format_input, inference, {("", "receivers")})
        self.add_flow(inference, decoder_configurator, {("transmitter", "in")})
        self.add_flow(decoder_configurator, inference_decoder, {("out", "receivers")})
        # point visualization
        self.add_flow(decoder_configurator, holoviz, {("point", "receivers")})
        self.add_flow(inference_decoder, postprocessor, {("transmitter", "in")})
        self.add_flow(postprocessor, holoviz, {("out", "receivers")})


if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser(description="Segment one thing application")
    parser.add_argument(
        "-s",
        "--source",
        choices=["v4l2"],
        default="v4l2",
        help=("If 'v4l2', uses the v4l2 device specified in the yaml file."),
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

    parser.add_argument(
        "-si", "--save_intermediate", action="store_true", help="Save intermediate tensors to disk"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output")
    args = parser.parse_args()

    if args.config == "none":
        config_file = os.path.join(os.path.dirname(__file__), "segment_one_thing.yaml")
    else:
        config_file = args.config

    app = SegmentOneThingApp(args.data, args.source, args.save_intermediate, args.verbose)
    app.config(config_file)
    app.run()
