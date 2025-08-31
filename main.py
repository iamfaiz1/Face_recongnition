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
import cv2



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



# Helper to load known images for display
def load_known_images():
    images = {}
    for fname in os.listdir(KNOWN_DIR):
        path = os.path.join(KNOWN_DIR, fname)
        name, _ = os.path.splitext(fname)
        try:
            img = Image.open(path).convert('RGB')
            img = img.resize((150, 150))
            images[name] = np.array(img)
        except:
            pass
    return images



def get_face_embedding(img_path):
    img = Image.open(img_path).convert('RGB')
    face = mtcnn(img)
    if face is None:
        return None
    face = face.unsqueeze(0).to(device)
    emb = resnet(face)
    return emb.detach().cpu().numpy()[0]



def get_face_embedding_from_pil(pil_img):
    face = mtcnn(pil_img)
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



def live_cam(known_embs, known_images):
    cap = cv2.VideoCapture(0)
    print("Starting webcam. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        boxes, probs = mtcnn.detect(pil_img)
        display_frame = frame.copy()
        matched_faces = []
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                face_crop = pil_img.crop((x1, y1, x2, y2)).resize((160, 160))
                emb = get_face_embedding_from_pil(face_crop)
                if emb is not None:
                    match, score = match_face(emb, known_embs)
                    if match:
                        label = f"{match} ({score:.2f})"
                        color = (0, 255, 0)
                        matched_faces.append((match, x1, y1, x2, y2))
                    else:
                        label = "Unknown"
                        color = (0, 0, 255)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
        # Show matched images beside
        if matched_faces:
            side_panel_width = 180
            side_img = np.ones((display_frame.shape[0], side_panel_width, 3), dtype=np.uint8) * 255
            y_offset = 10
            for match, x1, y1, x2, y2 in matched_faces:
                if match in known_images:
                    img = known_images[match]
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    h, w, _ = img.shape

                    # Ensure image fits side panel
                    if h > 150 or w > 150:
                        img = cv2.resize(img, (150, 150))
                        h, w, _ = img.shape
                    if y_offset + h < side_img.shape[0]:
                        side_img[y_offset:y_offset+h, 15:15+w] = img
                        cv2.putText(side_img, match, (15, y_offset+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
                        y_offset += h + 40
                        
            display_frame = np.hstack([display_frame, side_img])
        cv2.imshow('Facial Recognition - Live', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break
    cap.release()
    cv2.destroyAllWindows()



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
    # print("\nChecking images in 'to_check'...")
    # for fname in os.listdir(CHECK_DIR):
    #     path = os.path.join(CHECK_DIR, fname)
    #     emb = get_face_embedding(path)
    #     if emb is None:
    #         print(f"No face detected in {fname}")
    #         continue
    #     match, score = match_face(emb, known_embs)
    #     if match:
    #         print(f"{fname}: MATCHED with {match} (distance={score:.3f})")
    #     else:
    #         print(f"{fname}: No match found (distance={score:.3f})")

    print("Starting live camera... for face recognition")
    known_images = load_known_images()
    live_cam(known_embs, known_images)

if __name__ == "__main__":
    main()
