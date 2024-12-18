import cv2
import streamlit as st
import numpy as np
import os

# Charger le classificateur de cascade pour la détection de visages
file_path = 'C:/Users/DELL/Desktop/detection_visage/haarcascade_frontalface_default.xml'
if not os.path.isfile(file_path):
    st.error(f"Erreur : le fichier {file_path} n'existe pas.")
else:
    face_cascade = cv2.CascadeClassifier(file_path)
    if face_cascade.empty():
        st.error("Erreur : le classificateur de cascade n'a pas pu être chargé. Vérifiez le chemin du fichier.")

def detect_faces(selected_color, min_neighbors, scale_factor):
    # Initialiser la webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Erreur lors de l'accès à la webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

        # Convertir la couleur sélectionnée en tuple BGR
        color = tuple(int(selected_color[i:i + 2], 16) for i in (1, 3, 5))[::-1]

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Convertir l'image BGR à RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Afficher l'image dans Streamlit
        st.image(frame, channels="RGB", use_column_width=True)

        # Vérifier si la détection doit être arrêtée
        if st.session_state.get('stop_detection', False):
            break

    cap.release()
    st.session_state.stop_detection = False  # Réinitialiser l'état après la détection

def app():
    st.title("Détection de Visage avec l'Algorithme Viola-Jones")
    st.write("Bienvenue dans l'application de détection de visage !")

    # Instructions pour l'utilisateur
    st.markdown("""
    ### Instructions d'utilisation :
    1. Assurez-vous que votre webcam est correctement connectée et fonctionnelle.
    2. Cliquez sur le bouton **"Detect Faces"** ci-dessous pour commencer la détection de visages.
    3. Une fenêtre s'ouvrira affichant le flux de votre webcam avec des rectangles autour des visages détectés.
    4. Pour arrêter la détection, cliquez sur le bouton **"Stop Detection"**.
    """)

    selected_color = st.color_picker("Choisissez la couleur des rectangles", "#00FF00")
    min_neighbors = st.slider("Ajustez le paramètre minNeighbors", min_value=1, max_value=10, value=5)
    scale_factor = st.slider("Ajustez le paramètre scaleFactor", min_value=1.1, max_value=2.0, value=1.3, step=0.1)

    # Initialiser l'état de détection
    if 'stop_detection' not in st.session_state:
        st.session_state.stop_detection = False

    if st.button("Detect Faces"):
        st.session_state.stop_detection = False  # Réinitialiser l'état
        detect_faces(selected_color, min_neighbors, scale_factor)

    # Afficher le bouton "Stop Detection" uniquement si la détection est en cours
    if st.session_state.stop_detection:
        if st.button("Stop Detection"):
            st.session_state.stop_detection = True

if __name__ == "__main__":
    app()