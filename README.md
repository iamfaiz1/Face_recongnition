# Face_Recognition

## This project is made using OpenCV, FaceNet, Mtcnn.
It detects the Face and then checks from the previously loaded database of known Faces and matches the Face.

## how to run???
**step-1**
- clone this repo wherever u want.
- enter the Face_recognisiton using     *cd face_recognition*     then hit *tab* then *enter*
- **now u need to create a virtual environment to smoothly run this project**
    -now type:    *py -3.9 -m venv .venv*
    - activate the virtual environment:    *.venv\Scripts\activate*    (this commonad is only for windows user)
    - **download all dependencies:-**
    -  u just have to type in terminal (powershell):   *pip install -r requirements.txt*
    -  after installation u need to **create folder** :    *'known_faces'* here well add those faces which we will use for matching.
    -  now run the main fine in .venv environment :    *python main.py*
    -  and enjoyy.
