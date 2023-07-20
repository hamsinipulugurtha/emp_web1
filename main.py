from fastapi import FastAPI, UploadFile, File, Request,Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Annotated
from datetime import date, datetime
import json
import cv2
import face_recognition
import os
import numpy as np
from matplotlib import pyplot as plt
from google.cloud import storage
import cv2


import tensorflow as tf
from tensorflow import keras
import io
from PIL import Image
import numpy as np
import pandas as pd
import pickle
# import matplotlib.pyplot as plt
from code import download_blob
import json
from google.cloud import bigquery, storage
from google.oauth2 import service_account

from fastapi.responses import HTMLResponse
import pandas as pd
import os

count = 0
frame_counter = 0
attendance_dict = {}  # Dictionary to store attendance data
timestamps=[]
square_size = 500

def extract(request: Request):
    download_blob(bucket_name, source_file_name, dest_filename)

def list_images(bucket_name):
    blobs = client.list_blobs(bucket_name)
    images = []
    for blob in blobs:
        image_path = download_blob(bucket_name, blob.name, blob.name)
        images.append(image_path)
    return images


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get('/')  
def index(request : Request):
    context={"request" : request,
             "predictedtopic":"No Video"}
    return templates.TemplateResponse("index.html",context) 

@app.get("/main", response_class=HTMLResponse)
def lis( request : Request):
    images = list_images(bucket_name)  
    #print(images)
    context = {"request": request, "images": images}
    return templates.TemplateResponse("index.html", context)    

@app.post("/upload_video", response_class=HTMLResponse)
async def upload_video(request : Request, video_file: UploadFile = File(...)):
    b=recognize_faces()
    context = {
        "request": request, 
        "b": b
    }
    return templates.TemplateResponse("index.html",context)

with open('model.pkl', 'rb') as f:
    known_faces, known_names = pickle.load(f)

def recognize_faces():
    cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_x = (width - min(width, height)) // 2
    crop_y = (height - min(width, height)) // 2
    crop_width = min(width, height)
    crop_height = min(width, height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cropped_frame = frame[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]

        # Resize the square portion to the desired square frame size
        frame = cv2.resize(cropped_frame, (square_size, square_size))

        # Find faces in the frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        if len(face_locations) == 0:
            # Skip the frame if no faces are detected
            continue
        #timestamp
        if face_locations:
                adjusted_timestamp = datetime.now()
                timestamps.append(adjusted_timestamp)
        # Iterate over each detected face
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare face encoding with the known faces
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"

            # Find the best match
            if len(matches) > 0:
                face_distances = face_recognition.face_distance(known_faces, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]

                    # Update attendance dictionary with name and timestamp
                    # attendance_dict[name] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Draw a box around the face and label the name
                if face_locations:
                    adjusted_timestamp = datetime.now()
                    attendance_dict[name] = adjusted_timestamp.strftime("%Y-%B-%d %H:%M:%S")
                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, str(adjusted_timestamp.strftime("%Y-%B-%d %H:%M:%S")), (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Write the frame to the output video
        #out.write(frame)

        # Display the resulting frame
        # cv2.imshow(frame)
        #print(count)

    cap.release()
    #out.release()
    cv2.destroyAllWindows()
    return attendance_dict   




 