import os
import cv2
from app.face_recognition import faceRecognitionPipeline
from flask import render_template, request
import matplotlib.image as matimg

UPLOAD_FOLDER = 'static/upload'


def index():
    return render_template('index.html')


def app():
    return render_template('app.html')


def genderapp():
    if request.method == 'POST':
        f = request.files['image_name']
        filename = f.filename
        # save img in upload folder
        path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(path)
        pred_img, pred_dict = faceRecognitionPipeline(path)
        pred_filename = 'prediction_image.jpg'
        cv2.imwrite(f'./static/predict/{pred_filename}', pred_img)
        report = []
        for i, obj in enumerate(pred_dict):
            gray_image = obj['roi']
            eigen_image = obj['eig_img'].reshape((100, 100))
            gender_name = obj['prediction_name']
            score = round(obj['score']*100, 2)

            # save gray and eigen img in predict folder
            gray_image_name = f'roi_{i}.jpg'
            eigen_image_name = f'eigen_{i}.jpg'
            matimg.imsave(f'./static/predict/{gray_image_name}',
                          gray_image, cmap='gray')
            matimg.imsave(f'./static/predict/{eigen_image_name}',
                          eigen_image, cmap='gray')

            # save the report
            report.append(
                [gray_image_name, eigen_image_name, gender_name, score])

        return render_template('gender.html', fileupload=True, report = report)

    return render_template('gender.html', fileupload = False)
