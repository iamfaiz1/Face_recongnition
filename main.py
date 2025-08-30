"""
Facial Recognition Project using facenet-pytorch
Usage:
1. Place known people's images in the 'known_faces' folder (filename = person's name).
2. Place images to check in the 'to_check' folder.
3. Run this script to see matches.
"""
import os
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

# Directories
KNOWN_DIR = 'known_faces'
CHECK_DIR = 'to_check'

# Create folders if not exist
os.makedirs(KNOWN_DIR, exist_ok=True)
os.makedirs(CHECK_DIR, exist_ok=True)

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Models
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_face_embedding(img_path):
    img = Image.open(img_path).convert('RGB')
    face = mtcnn(img)
    if face is None:
        return None
    face = face.unsqueeze(0).to(device)
    emb = resnet(face)
    return emb.detach().cpu().numpy()[0]

def build_known_embeddings():
    known_embs = {}
    for fname in os.listdir(KNOWN_DIR):
        path = os.path.join(KNOWN_DIR, fname)
        name, _ = os.path.splitext(fname)
        emb = get_face_embedding(path)
        if emb is not None:
            known_embs[name] = emb
        else:
            print(f"No face detected in {fname}")
    return known_embs

def match_face(emb, known_embs, threshold=0.8):
    best_match = None
    best_score = float('inf')
    for name, kemb in known_embs.items():
        dist = np.linalg.norm(emb - kemb)
        if dist < best_score:
            best_score = dist
            best_match = name
    if best_score < threshold:
        return best_match, best_score
    else:
        return None, best_score

def main():
    print("Building known faces database...")
    known_embs = build_known_embeddings()
    print(f"Loaded {len(known_embs)} known faces.")
    print("\nChecking images in 'to_check'...")
    for fname in os.listdir(CHECK_DIR):
        path = os.path.join(CHECK_DIR, fname)
        emb = get_face_embedding(path)
        if emb is None:
            print(f"No face detected in {fname}")
            continue
        match, score = match_face(emb, known_embs)
        if match:
            print(f"{fname}: MATCHED with {match} (distance={score:.3f})")
        else:
            print(f"{fname}: No match found (distance={score:.3f})")

if __name__ == "__main__":
    main()
