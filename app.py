import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import numpy as np

# Cargar modelo y clasificador de rostros
modelo = load_model("modelo_emociones.h5")
emociones = ["Ira", "Asco", "Miedo", "Felicidad", "Tristeza", "Sorpresa", "Neutral"]
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


class AppEmociones:
	def __init__(self, root):
		self.root = root
		self.root.title("Detector de Emociones")

		self.video = cv2.VideoCapture(0)
		self.label = ttk.Label(root)
		self.label.pack()

		self.emocion_label = ttk.Label(root, text="Emoción: Ninguna")
		self.emocion_label.pack()

		self.actualizar_video()

	def actualizar_video(self):
		ret, frame = self.video.read()
		if ret:
			gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			rostros = face_cascade.detectMultiScale(gris, 1.3, 5)

			for (x, y, w, h) in rostros:
				rostro = gris[y:y + h, x:x + w]
				rostro = cv2.resize(rostro, (48, 48))
				rostro = rostro / 255.0
				rostro = np.expand_dims(rostro, axis=(0, -1))

				prediccion = modelo.predict(rostro)
				emocion = emociones[np.argmax(prediccion)]

				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
				self.emocion_label.config(text=f"Emoción: {emocion}")

			# Convertir frame a formato Tkinter
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			img = Image.fromarray(frame)
			imgtk = ImageTk.PhotoImage(image=img)
			self.label.imgtk = imgtk
			self.label.config(image=imgtk)

		self.root.after(10, self.actualizar_video)


# Iniciar la app
root = tk.Tk()
app = AppEmociones(root)
root.mainloop()