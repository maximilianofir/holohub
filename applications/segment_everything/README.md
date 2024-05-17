# Segment Everything App


Body pose estimation is a computer vision task that involves recognizing specific points on the human body in images or videos.
A model is used to infer the locations of keypoints from the source video which is then rendered by the visualizer. 

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
