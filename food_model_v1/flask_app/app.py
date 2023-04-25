from flask import Flask, render_template, request, session
import os
import subprocess
from werkzeug.utils import secure_filename
 
#*** Backend operation
 
# WSGI Application
# Defining upload folder path
UPLOAD_FOLDER = os.path.join('statics', 'uploads')
PREDICTION_FOLDER = os.path.join('statics', 'inferenced_imgs')

# # Define allowed files
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
 
# Provide template folder name
# The default folder name should be "templates" else need to mention custom folder name for template path
# The default folder name for static files should be "static" else need to mention custom folder for static path
app = Flask(__name__, template_folder='templates', static_folder='statics')
# Configure upload folder for Flask application
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICTION_FOLDER'] = PREDICTION_FOLDER
 
# Define secret key to enable session
app.secret_key = 'ABC'
app.config['SECRET_KEY'] = "ABC"
 
 
@app.route('/')
def index():
    return render_template('index_upload_and_display_image.html')
 
@app.route('/',  methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        # Upload file flask
        uploaded_img = request.files['uploaded-file']
        # Extracting uploaded data file name
        img_filename = secure_filename(uploaded_img.filename)
        # Upload file to database (defined uploaded folder in static path)
        uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
        # Storing uploaded file path in flask session
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
 
        return render_template('index_upload_and_display_image_page2.html')
 
@app.route('/show_image')
def displayImage():
    subprocess.run(['python3', '/Users/annieyuchuan/Desktop/USF/Spring2_project/food_model_v1/yolov5/detect.py', 
                '--source', '/Users/annieyuchuan/Desktop/USF/Spring2_project/food_model_v1/flask_app/statics/uploads',
                '--weights', '/Users/annieyuchuan/Desktop/USF/Spring2_project/food_model_v1/yolov5/runs/train/exp2/weights/best.pt',
                '--project', '/Users/annieyuchuan/Desktop/USF/Spring2_project/food_model_v1/flask_app/statics',
                '--name', 'inferenced_imgs'
                ])
    
    # Retrieving uploaded file path from session
    #img_file_path = session.get('uploaded_img_file_path', None)

    # Display image in Flask application web page
    infer_img = os.path.join(app.config['PREDICTION_FOLDER'], 'sample.jpg')
    return render_template('show_image.html', user_image = infer_img)
 
 
if __name__=='__main__':
    app.run(debug = True, port=9050)