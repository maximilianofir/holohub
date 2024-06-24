# Segment Everything App


Segment anything allows computing masks for objects in natural images. In this app it is applied to a videostream.

## Model

This application uses SAM models from https://github.com/facebookresearch/segment-anything/tree/main#model-checkpoints

## Requirements

This application uses a v4l2 compatible device as input.  Please plug in your input device (e.g., webcam) and ensure that `applications/segment_everything/segment_everything.yaml` is set to the corresponding device.  The default input device is set to `/dev/video0`.

## Run Instructions

Run the following commands to start the body pose estimation application:
```sh
./dev_container build --docker_file applications/segment_everything/Dockerfile --img holohub:sam
./dev_container launch --img holohub:sam
./run build segment_everything
./run launch segment_everything
```

## errors 
Error, not all input dimensions specified.
``` sh
[error] [holoinfer_constants.hpp:78] Inference manager, Error in inference setup: Error in Inference Manager, Sub-module->Setting Inference parameters: Error, not all input dimensions specified.

```
