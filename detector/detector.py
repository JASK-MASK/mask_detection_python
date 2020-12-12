from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np 
import imutils
import time
import cv2
import os
#Se importan las librerias a utilizar
from datetime import datetime
#=====================================FUNCIONES=========================================#

#se crea una funcion que recibe tres parametros, entre ellos el frame de video, los modelos preentrenados
#de deteccion de rostros y el modelo entrenado con el dataset previamente procesado
def predice_mascarilla(frame, faceNet, maskNet):

#Se crea un par ordenador que incluye la dimension del frame de video en formato bidimensional
    (h,w) = frame.shape[:2]
#Se crea un blob o por asi decirlo, una mancha de entrada en la cual el caffe model es capaz de detectar el rostro o rostros presentes
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224,224),(104.0, 177.0, 123.0))


    faceNet.setInput(blob)
    detections = faceNet.forward()
    
# se crean tres listas que reciben las caras, las ubicaciones de las caras y las predicciones respectivas

    faces = []
    locs = []
    preds = []
#en este caso se toma en cuenta un numero especifico de detecciones y se iteran para poder definir cada blob en el frame
    for i in range(0, detections.shape[2]):

#esta parte del codigo es tomada de un codigo base estandarizado para el uso de los modelos preentrenados dados a conocer por 
#el movimiento open source
        confidence = detections[0,0,i,2]
# un numero muy conocido dentro del caffe model es el porcentaje de confianza del modelo, y es en este punto donde se evalua con la estructura
# de control if
        if confidence > 0.5:

#Se convierte esta lista en un arreglo de valores especificados por el codigo ejemplar dentro de caffe model
            box = detections[0,0,i,3:7]*np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX,startY)=(max(0, startX), max(0, startY))
            (endX, endY) = (min(w-1,endX), min(h-1,endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224,224))
            face = img_to_array(face)
            face = preprocess_input(face)

#En esta seccion de codigo se crea el rectangulo pero es necesario destacar que unicamente se engloba la deteccion del rostro
            faces.append(face)
            locs.append((startX,startY,endX,endY))

    if len(faces) > 0:

        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32) #en esta linea de codigo es el punto donde se utiliza la prediccion del modelo entrenado
    return(locs,preds) #se devuelven las listas de las ubicaciones de los rostros y las predicciones de la presencia o ausencia de mascarilla

        label = "Mascarilla" if mask > withoutMask else "Sin Mascaril
prototxPath = "/home/alejandro/Escritorio/gui_app/face_detector/deploy.prototxt.txt"
weightsPath = "/home/alejandro/Escritorio/gui_app/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxPath, weightsPath)

maskNet = load_model("model-012.model")

#==============================================================================#

print("Empezando stream de video")

vs = VideoStream(src=0).start()

#==================================Activacion de camara===========================================#

while True:

    frame = vs.read() #se lee la camara integrada del dispositivo
    frame = imutils.resize(frame, width=400) #abrir la ventana de lo que detecta la camara con una dimension 400,400
    cv2.putText(frame,str(datetime.now()),(140,280), cv2.FONT_HERSHEY_PLAIN, 1,(255,255,255),2,cv2.LINE_AA) 
    #se coloca texto informativo como la fecha, la posicion, el tipo de fuente, el color y el grosor de la linea

    (locs, preds) = predice_mascarilla(frame, faceNet, maskNet) #se llama a la funcion de predictor de mascarilla para prepararse a 
    #ubicar en camara todas las posibles predicciones

#Se crea una tupla asociada a la union de las ubicaciones y las predicciones ejecutadas durante la proyeccion de video en el dispositivo
    for(box, pred) in zip(locs, preds):

        (startX, startY, endX, endY) = box #se dimensiona la region de interes
        (mask, withoutMask) = pred # se agregan las clases o categorias de cada predicciones

        label = "Mascarilla" if mask > withoutMask else "Sin Mascarilla" #se crean las etiquetas que se imprimiran sobre el frame de video
        color = (0,255,0) if label == "Mascarilla" else (0,0,255) # se escriben las palabras asociadas a cada clase creada

        

        cv2.putText(frame, label, (startX, startY-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2) #se crea el comando del texto por imprimir
        cv2.rectangle(frame, (startX, startY),(endX,endY), color, 2) #se crea el rectangulo sobre el cual se escribira la etiqueta
    
    cv2.imshow("Detector", frame)  #se muestra el frame de video con el nombre detector

    key = cv2.waitKey(1)

    if key == 27: #para salir presione ESC
        break

vs.stream.release() # se libera el frame de video
cv2.destroyAllWindows() # se destruyen las ventanas