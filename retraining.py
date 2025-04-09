import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from multiprocessing import Pool

# Configurar para usar mejor la memoria RAM disponible
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Directorios y parámetros
base_dir = "dataset/"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
emociones = ["Ira", "Asco", "Miedo", "Felicidad", "Tristeza", "Sorpresa", "Neutral"]
img_size = (48, 48)
batch_size = 128

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2
)


# Función separada para procesar imágenes
def procesar_imagen(args):
    emocion_idx, carpeta, archivo = args
    ruta = os.path.join(carpeta, archivo)
    imagen = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if imagen is not None:
        imagen = cv2.resize(imagen, img_size)
        imagen = img_to_array(imagen) / 255.0
        return (imagen, emocion_idx)
    return None


# Función para cargar imágenes
def cargar_imagenes(directorio):
    imagenes = []
    etiquetas = []

    tareas = []
    for idx, emocion in enumerate(emociones):
        carpeta = os.path.join(directorio, emocion)
        if not os.path.exists(carpeta):
            print(f"Advertencia: El directorio {carpeta} no existe")
            continue
        for archivo in os.listdir(carpeta):
            tareas.append((idx, carpeta, archivo))

    if not tareas:
        raise ValueError("No se encontraron imágenes en los directorios especificados")

    with Pool() as pool:
        resultados = pool.map(procesar_imagen, tareas)

    for resultado in resultados:
        if resultado is not None:
            imagen, etiqueta = resultado
            imagenes.append(imagen)
            etiquetas.append(etiqueta)

    return np.array(imagenes), np.array(etiquetas)


# Función principal para reentrenar el modelo
def reentrenar_modelo():
    print("Cargando datos...")
    try:
        X_train, y_train = cargar_imagenes(train_dir)
        X_val, y_val = cargar_imagenes(val_dir)
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return

    print("Datos de entrenamiento:", X_train.shape)
    print("Datos de validación:", X_val.shape)

    # Preparar etiquetas
    y_train = to_categorical(y_train, num_classes=len(emociones))
    y_val = to_categorical(y_val, num_classes=len(emociones))

    # Cargar el modelo preentrenado
    modelo = load_model("modelos entrenados/modelo_emociones_reentrenado.h5")
    print("Modelo preentrenado cargado exitosamente")

    # Opcional: Congelar capas iniciales para fine-tuning (ajusta según necesidad)
    # Por ejemplo, congelar todas las capas excepto las últimas densas
    for layer in modelo.layers[:-4]:  # Congela todas menos las últimas 4 capas
        layer.trainable = False
    for layer in modelo.layers[-4:]:  # Deja entrenables las últimas 4 capas
        layer.trainable = True

    # Compilar el modelo nuevamente (necesario tras cambiar trainable)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0002)  # Tasa más baja para reentrenamiento
    modelo.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # Resumen del modelo para verificar capas entrenables
    modelo.summary()

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

    # Reentrenar el modelo
    historial = modelo.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        epochs=1000,  # Menos épocas para reentrenamiento, ajusta según necesidad
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr]
    )

    # Guardar el modelo reentrenado
    modelo.save("modelo_emociones_reentrenado2.h5")
    print("Modelo reentrenado guardado como 'modelo_emociones_reentrenado.h5'")

    # Evaluar el modelo reentrenado
    loss, accuracy = modelo.evaluate(X_val, y_val)
    print(f"Precisión en validación tras reentrenamiento: {accuracy * 100:.2f}%")


if __name__ == '__main__':
    reentrenar_modelo()