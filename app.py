import streamlit as st
import easyocr
import cv2
import re
import os
from datetime import datetime
from spellchecker import SpellChecker
import numpy as np

# Regex for word tokens
WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ'-]+")

def tokenize_words(text: str):
    return WORD_RE.findall(text)

def extract_text_with_boxes(image_path):
    reader = easyocr.Reader(["en"], gpu=False)
    results = reader.readtext(image_path)
    return results  # [(bbox, text, confidence), ...]

def spellcheck_text(text):
    spell = SpellChecker(distance=2)
    tokens = tokenize_words(text.lower())
    misspelled = spell.unknown(tokens)

    corrections = {}
    corrected_tokens = []

    for word in tokens:
        if word in misspelled:
            correction = spell.correction(word)
            if correction is None:
                correction = word
            corrections[word] = correction
            corrected_tokens.append(correction)
        else:
            corrected_tokens.append(word)

    return corrections, " ".join(corrected_tokens)

def annotate_image_word_level(image_np, ocr_results, corrections):
    image = image_np.copy()

    for bbox, text, conf in ocr_results:
        words = text.split()
        num_words = len(words)
        x_min = min([p[0] for p in bbox])
        x_max = max([p[0] for p in bbox])
        y_min = min([p[1] for p in bbox])
        y_max = max([p[1] for p in bbox])
        word_width = (x_max - x_min) / max(1, num_words)

        for i, word in enumerate(words):
            word_clean = word.lower()
            word_bbox_top_left = (int(x_min + i*word_width), int(y_min))
            word_bbox_bottom_right = (int(x_min + (i+1)*word_width), int(y_max))

            if word_clean in corrections:
                cv2.rectangle(image, word_bbox_top_left, word_bbox_bottom_right, (0,0,255), 2)
                suggestion = f"{word} → {corrections[word_clean]}"
                text_pos = (word_bbox_top_left[0], max(0, word_bbox_top_left[1]-5))
                cv2.putText(image, suggestion, text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
    return image

# --- Streamlit UI ---
st.title("📑 OCR + Spellchecker Portal")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Save uploaded image temporarily
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_file = f"temp_{timestamp}.jpg"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # OCR
    ocr_results = extract_text_with_boxes(temp_file)
    full_text = "\n".join([res[1] for res in ocr_results])

    # Spellcheck
    corrections, corrected_text = spellcheck_text(full_text)

    # Annotated image
    image_np = cv2.imread(temp_file)
    annotated_image = annotate_image_word_level(image_np, ocr_results, corrections)

    # Display results
    st.subheader("🔍 Raw OCR Text")
    st.text(full_text)

    st.subheader("✅ Corrected Text")
    st.text(corrected_text)

    st.subheader("✏️ Corrections Made")
    if corrections:
        for wrong, right in corrections.items():
            st.write(f"**{wrong}** → {right}")
    else:
        st.write("No spelling issues found!")

    # Show annotated image
    st.subheader("🖼️ Annotated Image")
    st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), use_container_width=True)

    # Download corrected text as file
    corrected_file = f"corrected_text_{timestamp}.txt"
    with open(corrected_file, "w", encoding="utf-8") as f:
        f.write("Corrected Full Text:\n")
        f.write(corrected_text + "\n\n")
        f.write("Misspelled Words and Suggestions:\n")
        for wrong, right in corrections.items():
            f.write(f"{wrong} → {right}\n")

    with open(corrected_file, "rb") as f:
        st.download_button("⬇️ Download Corrected Text File", f, file_name=corrected_file, mime="text/plain")
