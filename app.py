import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
from sketch_transformer import sketch_transformer
from preprocess import image_to_stroke, stroke_to_sequence, pad_and_mask
import cv2
from PIL import Image
import numpy as np

MAX_LEN = 1500
LABEL_MAP = {0: 'apple', 1: 'banana', 2: 'basketball', 3: 'book', 4: 'car', 5: 'door', 6: 'eye', 7: 'smiley face', 8: 'sun', 9: 'tree', 10: 'zigzag'}

device = torch.device('cuda')
model = sketch_transformer(num_classes=11).to(device)
state_dict = torch.load('best_model.pth', map_location=device)
model.load_state_dict(state_dict)
model.eval()

st.title('Sketch Recognition')
st.markdown('Draw anything from the 11 categories on the canvas and let the model guess what it is!')
st.markdown('apple, banana, basketball, book, car, door, eye, smileyface, sun, tree, zigzag')

canvas_result = st_canvas(fill_color='rgba(255, 165, 0, 0.3)', stroke_width=4, stroke_color='#000000', background_color='#ffffff', height=280, width=280, drawing_mode='freedraw', key='canvas')

if st.button('Predict') and canvas_result.image_data is not None:
    image = canvas_result.image_data.astype(np.uint8)
    strokes = image_to_stroke(image)

    if not strokes:
        st.warning('Please draw something before clicking on predict!')

    else:
        sequence = stroke_to_sequence(strokes)
        padded, mask = pad_and_mask(sequence, max_len=MAX_LEN)
        padded = padded.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(padded, mask)
            probs = torch.softmax(output, dim=1)[0]
            topk = torch.topk(probs, 3)

        st.success(f"**Top Prediction:** {LABEL_MAP[topk.indices[0].item()]} ({topk.values[0].item()*100:.2f}%)")

        st.subheader("Top 3 Predictions:")
        for i in range(3):
            st.write(f"{LABEL_MAP[topk.indices[i].item()]} — {topk.values[i].item()*100:.2f}%")

if st.button('Redraw'):
    if 'canvas' in st.session_state:
        del st.session_state['canvas']
    st.rerun()