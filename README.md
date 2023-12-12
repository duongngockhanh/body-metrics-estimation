# Height Measurement

```commandline
cd Combine
uvicorn app:app --reload
```

## 1. Fix bug
### 1.1. freenect library: [discussion on github](https://github.com/OpenKinect/libfreenect/issues/550) - [solution](https://naman5.wordpress.com/2014/06/24/experimenting-with-kinect-using-opencv-python-and-open-kinect-libfreenect/)
```sudo freenect-glview``` instead of ```freenect-glview``` in the 7th step.
### 1.2. import detection2
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
### 1.3. import densepose
```
pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose
```