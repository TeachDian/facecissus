# After importing necessary modules make sure muna na naka install lahat yuung modules, pag may lines
# sa ilalim means hindi to install. so pwede mo i check with pip sa termninal. make sure din na meron
# kang python so you can check it using python --version sa terminal preferably Python 3.12.2
# pip install flask
# pip install joblib
# pip install pandas
# pip install scikit-learn
# pip install pip install face_recognition
# pip install numpy
# pip install opencv-python
import cv2
import os
import csv
from flask import Flask, render_template, request, redirect, url_for, session
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

# Defining Flask App
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Secret key for session management

# This dictionary simulates a simple user database.
# In a real application, you would use a database.

# Add this route to your Flask application
@app.route('/logout', methods=['POST'])
def logout():
    # Remove username from session
    session.pop('username', None)
    # Redirect user to the login page
    return redirect(url_for('login'))
# Function to check if a user exists in the CSV file
def user_exists(username):
    with open('users.csv', 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['username'] == username:
                return True
    return False

# Function to authenticate a user
def authenticate_user(username, password):
    with open('users.csv', 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['username'] == username and row['password'] == password:
                return True
    return False

# Registration route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check if username already exists
        if user_exists(username):
            error_message = "Username already exists. Please choose a different username."
            return render_template('register.html', error=error_message)
        # Check if username has at least 4 characters
        elif len(username) < 4:
            error_message = "Username must have at least 4 characters."
            return render_template('register.html', error=error_message)
        # Check if password has at least 8 characters
        elif len(password) < 8:
            error_message = "Password must have at least 8 characters."
            return render_template('register.html', error=error_message)
        else:
            # Write the new user data to the CSV file
            with open('users.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([username, password])
            return redirect(url_for('login'))
    else:
        # If request method is GET, render the registration form
        return render_template('register.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check if username and password match
        if authenticate_user(username, password):
            # Store username in session to indicate user is logged in
            session['username'] = username
            return redirect(url_for('home'))
        else:
            error_message = "Invalid username or password. Please try again."
            return render_template('login.html', error=error_message)
    else:
        # If request method is GET, render the login form
        return render_template('login.html')
    
# Create users.csv file if it doesn't exist
if not os.path.isfile('users.csv'):
    with open('users.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['username', 'password'])  # Write header row

# Face counter so the higher the value the more accurate ang problem is more storage and computing
# power yung needed. recommended is around 50 to 200 picture and if possible mas better camera 
# quality and needed.

nimgs = 200

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")
attendance_file_path = f'Attendance/Attendance-{datetoday2}.csv'

# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

# get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))

# extract the face from an image
def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []

# ang pogi ko
# Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

# A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')
    
# Add route to retrain the model
@app.route('/retrain')
def retrain_model():
    train_model()  # Call your train_model function to retrain the model
    return redirect('/')  # Redirect back to the home page after retraining

def extract_attendance():
    try:
        df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
        names = df['Name']
        rolls = df['Roll']
        times = df['Time']
        l = len(df)
        return names, rolls, times, l
    except FileNotFoundError:
        return [], [], [], 0

# Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')

## A function to get names and rol numbers of all users
def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l

## A function to delete a user folder 
def deletefolder(duser):
    pics = os.listdir(duser)
    for i in pics:
        os.remove(duser+'/'+i)
    os.rmdir(duser)

################## ROUTING FUNCTIONS #########################

# Home route
@app.route('/')
def home():
    # Check if user is logged in (session contains username)
    if 'username' in session:
        names, rolls, times, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, username=session['username'])
    else:
        # If user is not logged in, redirect to login page
        return redirect(url_for('login'))

## List users page
@app.route('/listusers')
def listusers():
    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


## Delete functionality
@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    duser = request.args.get('user')
    deletefolder('static/faces/'+duser)

    ## if all the face are deleted, delete the trained file...
    if os.listdir('static/faces/')==[]:
        os.remove('static/face_recognition_model.pkl')
    
    try:
        train_model()
    except:
        pass

    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# Add route to view today's attendance
@app.route('/attendance')
def view_attendance():
    names, rolls, times, l = extract_attendance()
    return render_template('attendance.html', names=names, rolls=rolls, times=times, l=l, datetoday2=datetoday2)

# Our main Face Recognition functionality. 
# This function will run when we click on Take Attendance Button.
# Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    prediction = model.predict(facearray)
    if prediction[0] in os.listdir('static/faces'):
        return prediction[0]  # Return the recognized face ID
    else:
        return "Unregistered"  # Return "Unregistered" if face is not found in the database

# Our main Face Recognition functionality. 
# This function will run when we click on Take Attendance Button.
@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(0)
    last_detected_person = None  # Initialize last detected person as None
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))
            
            # Check if identified_person is registered or not
            if identified_person != "Unregistered":
                last_detected_person = identified_person  # Update last detected person
                cv2.putText(frame, f'{identified_person}', (x+5, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                # If face is not registered, display as "Unregistered"
                cv2.putText(frame, 'Unregistered', (x+5, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    
    # Add attendance for the last detected person if recognized
    if last_detected_person is not None and last_detected_person != "Unregistered":
        add_attendance(last_detected_person)
        
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# A function to add a new user.
# This function will run when we add a new user.
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == nimgs*5:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# CSV reader side of things hahaha

# Function to get list of all CSV files
def get_csv_files():
    attendance_files = [file for file in os.listdir('Attendance') if file.endswith('.csv')]
    return attendance_files

# Function to read CSV file and extract data
def read_csv_file(filename):
    df = pd.read_csv(os.path.join('Attendance', filename))
    return df

# Function to delete CSV file
def delete_csv_file(filename):
    os.remove(os.path.join('Attendance', filename))

# Route to view all CSV files
@app.route('/view_csv_files')
def view_csv_files():
    csv_files = get_csv_files()
    return render_template('view_csv_files.html', csv_files=csv_files)

# Route to view and edit CSV file
@app.route('/view_csv_file/<filename>', methods=['GET', 'POST'])
def view_csv_file(filename):
    df = read_csv_file(filename)
    if request.method == 'POST':
        # Handle editing the CSV file here
        # For demonstration, let's just update the first row
        df.iloc[0, 0] = request.form['new_value']
        df.to_csv(os.path.join('Attendance', filename), index=False)
        return redirect(url_for('view_csv_files'))
    return render_template('view_csv_file.html', filename=filename, df=df)

# Route to delete CSV file s
@app.route('/delete_csv_file/<filename>')
def delete_csv_file_route(filename):
    delete_csv_file(filename)
    return redirect(url_for('view_csv_files'))

# Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)