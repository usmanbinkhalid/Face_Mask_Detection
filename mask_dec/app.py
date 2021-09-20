from flask import Flask, render_template, request
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
import cv2
from keras.models import load_model, model_from_json
import numpy as np

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after', methods=['GET', 'POST'])
def after():
    img = request.files['file1']

    img.save('file.jpg')

    ####################################

    img1 = cv2.imread('file.jpg')
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    faces = cascade.detectMultiScale(gray, 1.1, 3)

    for x,y,w,h in faces:
        cv2.rectangle(img1, (x,y), (x+w, y+h), (0,255,0), 2)

        cropped = img1[y:y+h, x:x+w,:]

    cv2.imwrite('after.jpg', img1)

    try:
        cv2.imwrite('cropped.jpg', cropped)

    except:
        pass

    #####################################

    try:
        image = cv2.imread('cropped.jpg', 0)
    except:
        image = cv2.imread('file.jpg', 0)




    face_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_frame = cv2.resize(face_frame, (224, 224))
    face_frame = img_to_array(face_frame)
    face_frame = np.expand_dims(face_frame, axis=0)
    face_frame =  preprocess_input(face_frame)

    model=load_model('mask_recog.h5')



    prediction = model.predict(face_frame)

    label_map =   ['Yes! You are keeping caution','No! Please wear a mask']
    prediction = np.argmax(prediction)

    final_prediction = label_map[prediction]

    return render_template('after.html', data=final_prediction)



if __name__ == "__main__":
    app.run()
