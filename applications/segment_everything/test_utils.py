import pytest
import numpy as np
import cupy as cp
import os
import tempfile
from utils import save_cupy_tensor, DecoderInputData, CupyArrayPainter


def test_save_cupy_tensor():
    # use a temporary directory from tempfile to save the tensor
    with tempfile.TemporaryDirectory() as temp_dir:
        # create a random tensor
        tensor = np.random.rand(10, 10)
        # save the tensor
        save_cupy_tensor(tensor, temp_dir, counter=0, word="test", verbose=False)
        # check if the tensor was saved
        numpy_filepath_expected = os.path.join(temp_dir, "test_0.npy")
        assert os.path.exists(numpy_filepath_expected)


class TestDecoderInputData:
    # test the constructor via the static method create_decoder_inputs_from
    def test_create_decoder_inputs_from(self):
        point = (250, 300)
        dtype = np.float32
        decoder_input = DecoderInputData.create_decoder_inputs_from(dtype=dtype, input_point=point)
        print(f"created inputs {decoder_input}")

        # check the attributes and their shapes. The dimensions must match the onnx model dimensions until dynamic
        # axes can be used in holoscan.
        assert decoder_input.has_mask_input.shape == (1.0, 1.0, 1.0, 1.0)
        assert decoder_input.image_embeddings is None
        assert decoder_input.mask_input.shape == (1, 1, 256, 256)
        assert decoder_input.orig_im_size is None
        assert decoder_input.point_coords.shape == (1, 2, 2)
        assert np.all(decoder_input.point_coords == np.array([[[250, 300], [0, 0]]], dtype=dtype))
        assert decoder_input.point_labels.shape == (1, 2)
        assert np.all(decoder_input.point_labels == np.array([[1, -1]], dtype=dtype))

    @pytest.mark.parametrize(
        "point, dtype, orig_height, orig_width, resized_height, resized_width, expected_coords",
        [
            (
                (500, 500),
                np.float32,
                1024,
                1024,
                1024,
                1024,
                np.array([[[500, 500], [0, 0]]], dtype=np.float32),
            ),
            (
                (500, 500),
                np.float32,
                1024,
                1024,
                512,
                512,
                np.array([[[250, 250], [0, 0]]], dtype=np.float32),
            ),
            (
                (500, 500),
                np.float32,
                1024,
                1024,
                2048,
                2048,
                np.array([[[1000, 1000], [0, 0]]], dtype=np.float32),
            ),
            (
                (500, 500),
                np.float32,
                1024,
                1024,
                1024,
                512,
                np.array([[[250, 500], [0, 0]]], dtype=np.float32),
            ),
        ],
    )
    def test_scale_coords_parametrized(
        self, point, dtype, orig_height, orig_width, resized_height, resized_width, expected_coords
    ):
        # test same size, halving, doubling and stretching as coordinate transformations
        decoder_input = DecoderInputData.create_decoder_inputs_from(dtype=dtype, input_point=point)
        scaled_coords = DecoderInputData.scale_coords(
            decoder_input.point_coords,
            orig_height=orig_height,
            orig_width=orig_width,
            resized_height=resized_height,
            resized_width=resized_width,
            dtype=dtype,
        )
        assert np.all(scaled_coords == expected_coords)


@pytest.fixture
def painter():
    return CupyArrayPainter()


@pytest.fixture
def random_data():
    return (cp.random.rand(10, 10) - 0.5) * 100


class TestCupyArrayPainter:
    def test_constructor(self, painter):
        # Check the attributes
        assert isinstance(painter, CupyArrayPainter)
        assert isinstance(painter.colormap, cp.ndarray)
        # Check that the colormap has 256 color values for rgba, eg the shape is (256, 4)
        assert painter.colormap.shape == (256, 4)

    def test_normalize_data(self, painter, random_data):
        # Normalize the data
        normalized_data = painter.normalize_data(random_data)
        # Check the normalized data
        assert normalized_data.max() <= 1
        assert normalized_data.min() >= 0

    def test_apply_colormap(self, painter, random_data):
        # Normalize the data
        normalized_data = painter.normalize_data(random_data)
        # Apply the colormap
        colored_data = painter.apply_colormap(normalized_data)
        # Check the colored data
        assert colored_data.shape == (10, 10, 4)
        assert colored_data.max() <= 255
        assert colored_data.min() >= 0
        assert isinstance(colored_data, cp.ndarray)

    def test_to_rgba(self, painter, random_data):
        # Convert to rgba
        rgba_data = painter.to_rgba(random_data)
        # Check the rgba data
        assert rgba_data.shape == (10, 10, 4)
        assert rgba_data.max() <= 255
        assert rgba_data.min() >= 0
        assert isinstance(rgba_data, cp.ndarray)
