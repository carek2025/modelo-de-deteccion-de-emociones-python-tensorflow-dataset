import cv2
from tensorflow.keras.models import load_model
import numpy as np

# Cargar el modelo entrenado
modelo = load_model("modelo_emociones.h5")
emociones = ["Ira", "Asco", "Miedo", "Felicidad", "Tristeza", "Sorpresa", "Neutral"]

# Cargar el clasificador de rostros de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Iniciar la cámara
cap = cv2.VideoCapture(0)

while True:
	ret, frame = cap.read()
	if not ret:
		break

	# Convertir a escala de grises
	gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rostros = face_cascade.detectMultiScale(gris, 1.3, 5)

	for (x, y, w, h) in rostros:
		# Extraer el rostro
		rostro = gris[y:y + h, x:x + w]
		rostro = cv2.resize(rostro, (48, 48))
		rostro = rostro / 255.0
		rostro = np.expand_dims(rostro, axis=(0, -1))  # Ajustar dimensiones

		# Predecir emoción
		prediccion = modelo.predict(rostro)
		emocion = emociones[np.argmax(prediccion)]

		# Dibujar rectángulo y etiqueta
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		cv2.putText(frame, emocion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

	# Mostrar el video
	cv2.imshow("Detección de Emociones", frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):  # Presiona 'q' para salir
		break

cap.release()
cv2.destroyAllWindows()