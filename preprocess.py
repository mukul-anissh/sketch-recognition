import numpy as np
import cv2
import torch

def image_to_stroke(image):
    if image is None:
        return []
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    strokes = []
    for cnt in contours:
        cnt = cnt.squeeze()
        if cnt.ndim != 2 or len(cnt) < 2:
            continue

        x = cnt[:, 0].tolist()
        y = cnt[:, 1].tolist()

        # Resample every few points to reduce noise (optional)
        stride = max(1, len(x) // 20)
        x = x[::stride]
        y = y[::stride]

        strokes.append([x, y])

    return strokes

def stroke_to_sequence(sketch):
    prev_x, prev_y = 0, 0
    sequence = []
    
    for stroke in sketch:
        x_list, y_list = stroke[0], stroke[1]

        for i in range(len(x_list)):
            x, y = x_list[i], y_list[i]
            dx, dy = x-prev_x, y-prev_y
            pen = 0

            if i == len(x_list)-1:
                pen = 1
            
            sequence.append([dx, dy, pen])
            prev_x, prev_y = x, y
    
    sequence.append([0, 0, 2])
    return torch.tensor(sequence, dtype=torch.float)

def pad_and_mask(sequence, max_len=1500):
    seq_len = sequence.shape[0]
    if seq_len > max_len:
        sequence = sequence[:max_len]
        seq_len = max_len

    pad_len = max_len - seq_len
    padded = torch.cat([sequence, torch.zeros(pad_len, 3)], dim=0)
    mask = torch.zeros(max_len, dtype=torch.int64)
    mask[:seq_len] = 1

    return padded, mask