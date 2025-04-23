from flask import Flask, render_template, Response, jsonify, request, session
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
import os
import cv2
import numpy as np
from YOLO_Video import video_detection  # Import YOLOv8 + DeepSORT processing function

app = Flask(__name__)

app.config['SECRET_KEY'] = 'blurshield'
app.config['UPLOAD_FOLDER'] = 'static/files'

# Form for file upload
class UploadFileForm(FlaskForm):
    file = FileField("Upload Video", validators=[InputRequired()])
    submit = SubmitField("Run")

# Improved function to generate video frames
def generate_frames(path_x=''):
    yolo_output = video_detection(path_x)

    for detection_ in yolo_output:
        # Ensure valid frame
        if detection_ is None or not isinstance(detection_, np.ndarray):
            print("❌ Error: Invalid frame received from YOLO!")
            continue  # Skip invalid frames

        ret, buffer = cv2.imencode('.jpg', detection_)
        if not ret:
            print("❌ Error: Could not encode frame!")
            continue

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    session.clear()
    return render_template('index.html')

@app.route("/webcam", methods=['GET', 'POST'])
def webcam():
    session.clear()
    return render_template('ui.html')

@app.route('/FrontPage', methods=['GET', 'POST'])
def front():
    form = UploadFileForm()

    if form.validate_on_submit():
        file = form.file.data
        filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(filepath)  # Save uploaded video file
        session['video_path'] = filepath  # Store video path in session

    return render_template('videoprojectnew.html', form=form)

@app.route('/video')
def video():
    video_path = session.get('video_path', None)

    if not video_path or not os.path.exists(video_path):
        print("❌ Error: Video file not found!")
        return "Error: No valid video file uploaded!", 404

    return Response(generate_frames(path_x=video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webapp')
def webapp():
    return Response(generate_frames(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
