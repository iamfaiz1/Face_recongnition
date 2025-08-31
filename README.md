# Face_Recognition

This project is made using **OpenCV**, **FaceNet**, and **MTCNN**.  
It detects a face and then checks from the previously loaded database of known faces to find a match.

---

## 🚀 How to Run

### Step 1: Clone the Repository at ur desired location, below is the command
```bash
git clone git@github.com:iamfaiz1/Face_recongnition.git

//enter the project directory (type then enter):
cd face_recognition


// creating virtual environment for smooth setup and executiion (make sure u have python 3.9 installed otherwises install it first)

py -3.9 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
mkdir known_faces
python main.py
