import os
import warnings
import wget
import argparse
from copy import deepcopy

import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry
from segment_anything.utils.onnx import SamOnnxModel




def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))


class ImageTensor:
    def __init__(self, image):
        self.image = image
        self.orig_width, self.orig_height = image.size
        self.resized_width, self.resized_height = None, None
        self.pad_width, self.pad_height = None, None

    def size(self):
        return self.image.size

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension
        """
        old_h, old_w = self.orig_height, self.orig_width
        new_h, new_w = self.resized_height, self.resized_width
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords


class ImagePreprocessor:
    def __init__(self, long_side_max=1024, mean=None, std=None, image_format="RGB", pad_to_square=True):
        self.long_side_max = long_side_max
        self.mean = mean
        self.std = std
        self.image_format = image_format
        self.pad_to_square = pad_to_square
        if self.mean is None:
            self.mean = np.array([123.675, 116.28, 103.53])
        if self.std is None:
            self.std = np.array([58.395, 57.12, 57.375])


    def resize_image_to_long_side(self, img: ImageTensor):
        if self.long_side_max is None:
            return img
        orig_width, orig_height = img.image.size
        if orig_width > orig_height:
            img.resized_width = self.long_side_max
            img.resized_height = int(self.long_side_max / orig_width * orig_height)
        else:
            img.resized_height = self.long_side_max
            img.resized_width = int(self.long_side_max / orig_height * orig_width)

        img.image = img.image.resize((img.resized_width, img.resized_height), Image.Resampling.BILINEAR)
        return img

    def make_image_rgb(self, image):
        if image.image.mode == "RGB":
            return image
        else:
            image.image = image.image.convert("RGB")
            return image

    def pad_image_to_square(self, image):
        if isinstance(image, ImageTensor):
            image.image = self.pad_image_to_square(image.image)
            return image
        else:
            h, w = image.shape[2:]
            max_dim = max(h, w)
            pad_h = max_dim - h
            pad_w = max_dim - w
            image = np.pad(image, ((0,0), (0,0), (0,pad_h), (0,pad_w)), mode="constant", constant_values=0)
            return image

    def normalize_image(self, image):
        if isinstance(image, ImageTensor):
            image.image = self.normalize_image(image.image)
            return image
        else:
            image = (image - self.mean) / self.std
            return image

    def to_tensor(self, image):
        if isinstance(image, ImageTensor):
            image.image = self.to_tensor(image.image)
            return image
        else:
            image = image.transpose(2,0,1)[None,:,:,:].astype(np.float32)
            return image

    def from_image_to_input(self, image):
        image = self.make_image_rgb(image)
        image = self.resize_image_to_long_side(image)
        image = self.normalize_image(image)
        image = self.to_tensor(image)
        # pad to square
        if self.pad_to_square:
            image = self.pad_image_to_square(image)
        return image


class ModelDownloader:
    def __init__(self, download_url:str=None) -> None:
        if download_url is None:
            download_url = R"https://dl.fbaipublicfiles.com/segment_anything/"
        self.models = {
            "sam_vit_h": download_url + "sam_vit_h_4b8939.pth",
            "sam_vit_b": download_url + "sam_vit_b_01ec64.pth",
            "sam_vit_l": download_url + "sam_vit_l_0b3195.pth",
        }

    def download_model(self, model_type, filepath:str=None):
        if filepath is None:
            # create a download folder in the current directory and use it to save the models there
            filedir = os.path.join(os.getcwd(), "downloads")
            os.makedirs(filedir, exist_ok=True)
            filepath = os.path.join(filedir, f"{model_type}.pth")
        if model_type not in self.models:
            raise ValueError("Invalid model_type")
        # only download the model if it doesn't already exist
        if os.path.exists(filepath):
            return filepath
        else:
            print(f"Downloading model: {model_type}")
            url = self.models[model_type]
            wget.download(url, filepath)

        return filepath



downloader = ModelDownloader()
for model_type in ["sam_vit_h", "sam_vit_b", "sam_vit_l"]:
    downloaded_filepath = downloader.download_model(model_type)
    print(f"Model downloaded to: {downloaded_filepath}")

for model_type in ["vit_h", "vit_b", "vit_l"]:
    checkpoint_path = os.path.join(os.getcwd(), "downloads", f"sam_{model_type}.pth")
    onnx_decoder_path = os.path.join(os.getcwd(), "onnx", f"sam_{model_type}_query_decoder.onnx")
    onnx_encoder_path = os.path.join(os.getcwd(), "onnx", f"sam_{model_type}_encoder.onnx")
    if not os.path.exists(os.path.dirname(onnx_decoder_path)):
        os.makedirs(os.path.dirname(onnx_decoder_path))

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)

    onnx_model = SamOnnxModel(sam, return_single_mask=True)

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
    }
    output_names = ["masks", "iou_predictions", "low_res_masks"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(onnx_decoder_path, "wb") as f:
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=17,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )

        with open(onnx_encoder_path, "wb") as f:
        # Export images encoder from SAM model to ONNX
            torch.onnx.export(
                f=f,
                model=sam.image_encoder,
                args=torch.randn(1, 3, 1024, 1024),
                input_names=["images"],
                output_names=["embeddings"],
                export_params=True
        )
