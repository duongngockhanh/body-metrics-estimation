# Height Measurement

```commandline
cd Combine
uvicorn app:app --reload
```

## Fix bug
1. freenect library: [discussion on github](https://github.com/OpenKinect/libfreenect/issues/550) - [solution](https://naman5.wordpress.com/2014/06/24/experimenting-with-kinect-using-opencv-python-and-open-kinect-libfreenect/)
2. import detection2:
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
3. import densepose:
```
pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose
```

## Requirements.txt
Because the project is too large for cloning, so I would take all the changes out in README.md.

```
uvicorn==0.24.0.post1
opencv-python==4.8.1.78
Pillow==10.1.0
fastapi==0.104.1
pydantic==2.5.2
numpy==1.26.2
pandas==2.1.4
torch==1.8.0
torchvision==0.9.0
Jinja2==3.1.2
```
