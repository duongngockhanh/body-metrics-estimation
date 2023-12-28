# Body Metrics Estimation System
```commandline
uvicorn app:app --reload
```


# Usage Guide
## 1. Overall measurements

- In this section, you can capture a frontal view photo to prepare for calculating body measurements by pressing buttons **Scan3D**, **Pose**, **Seg**, or **All**. **Scan3D** corresponds to Human Scan 3D, **Pose** to Human Pose, **Seg** to Human Segmentation, and **All** automatically performs all three tasks. Similarly, for the profile view, we have buttons **Scan3D90**, **Pose90**, **Seg90**, and **All90** with similar functionalities.

- Afterward, you can view the calculated parameters by pressing the **Send data** button.

- Additionally, our measurements rely on two parameters: d_param and f_param. In which, d_param represents the distance between the subject being measured and the camera, and f_param is the focal length of the camera, measured in centimeters. You can adjust these parameters by entering values into the respective textboxes and press the **Init Measure**. By default, d_param is set to 285 cm, and f_param is set to 650.1 cm.

![e1](https://github.com/duongngockhanh/height-measurement/assets/87640587/3f692405-8cf5-413f-9c9c-14dfd3d80062)

### Notice:
- In our experiment, the images used for measurements are selected from a default image. You can uncomment the command ```_, frame = cap.read()``` to directly capture images from the camera in **app.py**.

![e4](https://github.com/duongngockhanh/height-measurement/assets/87640587/f96778b8-f867-494e-9a4a-e47e93f9364b)

## 2. Show a few specified body measurements

You can press the **Show table** button to perform some specified calculations or measurements after completing several overall measurements.

![e2](https://github.com/duongngockhanh/height-measurement/assets/87640587/264a3e78-6254-4d5a-a6a0-e0907db43d18)

![e3](https://github.com/duongngockhanh/height-measurement/assets/87640587/f426c1db-1709-428e-8efe-76a0fcc7acce)

