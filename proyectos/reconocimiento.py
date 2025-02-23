import cv2
import face_recognition
import os
import numpy as np

ruta_imagenes = "proyectos\\imagenes"
known_faces = {}

if not os.path.exists(ruta_imagenes):
    raise FileNotFoundError(f"❌ No se encontró la carpeta '{ruta_imagenes}'. ¡Créala y agrega imágenes!")

# Cargar imágenes y obtener sus codificaciones de los rostros
for archivo in os.listdir(ruta_imagenes):
    if archivo.endswith((".jpg", ".jpeg", ".png")):
        ruta = os.path.join(ruta_imagenes, archivo)
        imagen = face_recognition.load_image_file(ruta)
        codificaciones = face_recognition.face_encodings(imagen)
        if codificaciones:
            known_faces[archivo.split(".")[0]] = codificaciones[0]
        else:
            print(f"⚠️ No se detectó un rostro en '{archivo}', ignorado.")

if not known_faces:
    raise ValueError("❌ No se encontraron rostros válidos en la carpeta 'imagenes'.")

video_capture = cv2.VideoCapture(0)

while True:
    # Capturar frame
    ret, frame = video_capture.read()
    if not ret:
        break

    # Reducir tamaño para mejorar rendimiento
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    
    # Convertir BGR a RGB correctamente
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    
    # Obtener codificaciones
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Comparar con imágenes conocidas
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        nombre = "Desconocido"
        color = (0, 0, 255)  

        for usuario, user_encoding in known_faces.items():
            resultado = face_recognition.compare_faces([user_encoding], face_encoding, tolerance=0.5)
            if resultado[0]:
                nombre = usuario
                color = (0, 255, 0)  
                break

        # Ajustar coordenadas aquiiii
        top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2
        
        # Dibujar rectángulo y nombre de mi
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, nombre, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        if nombre != "Desconocido":
            print(f"✅ Rostro reconocido: {nombre}.jpg")


    cv2.imshow('Reconocimiento Facial', frame)

    # Salir tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()