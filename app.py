import time
from flask import Flask, render_template, Response, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import datetime
import cv2
import mediapipe as mp
import math
import threading

import pygame

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///yoga.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)
    progress = db.relationship('Progress', backref='user', lazy=True)
    achievements = db.relationship('Achievement', backref='user', lazy=True)

class Progress(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    exercise_name = db.Column(db.String(100), nullable=False)
    count = db.Column(db.Integer, nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class Achievement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(500), nullable=False)
    date_achieved = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class Challenge(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(500), nullable=False)
    start_date = db.Column(db.DateTime, default=datetime.utcnow)
    end_date = db.Column(db.DateTime)
    progress = db.relationship('ChallengeProgress', backref='challenge', lazy=True)

class ChallengeProgress(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    challenge_id = db.Column(db.Integer, db.ForeignKey('challenge.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    progress = db.Column(db.Integer, nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ---------------------------------------------Pose Detection and Classification---------------------------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

pose_status = "Unknown"
frame_lock = threading.Lock()
current_frame = None

curl_count = 0
previous_label = None

def detectPose(image, pose):
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    height, width, _ = image.shape
    landmarks = []

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height), (landmark.z * width)))

    return output_image, landmarks

def classifyPose(landmarks):
    global beep_sound, curl_count, previous_label
    label = 'Unknown Pose'
    color = (0, 0, 255)

    if len(landmarks) < 33:  # Check if there are enough landmarks
        return label, color

    # Extract angles for different poses
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    
    shoulder_to_elbow_to_wrist_angle_left = calculateAngle(left_shoulder, left_elbow, left_wrist)
    shoulder_to_elbow_to_wrist_angle_right = calculateAngle(right_shoulder, right_elbow, right_wrist)
    
    # Biceps Curl Detection
    if shoulder_to_elbow_to_wrist_angle_left < 50:
        label = 'Biceps Curl'
        color = (0, 255, 0)
        beep_sound.play()
    
    # Check if the right hand angle is less than 50 degrees
    if shoulder_to_elbow_to_wrist_angle_right < 50:
        label = 'Biceps Curl'
        color = (0, 255, 0)
        beep_sound.play()
    
    if previous_label != 'Biceps Curl':
        curl_count += 1
        previous_label = 'Biceps Curl'
    
    # Tree Pose Detection
    if len(landmarks) >= 33:
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        left_leg_angle = calculateAngle(left_hip, left_knee, left_ankle)
        right_leg_angle = calculateAngle(right_hip, right_knee, right_ankle)

        if left_leg_angle > 165 and left_leg_angle < 195 and right_leg_angle > 30 and right_leg_angle < 60:
            label = 'Tree Pose'
            color = (0, 255, 0)
            beep_sound.play()
            previous_label = 'Tree Pose'

    # Pushup Detection
    elbow_angle = calculateAngle(left_shoulder, left_elbow, left_wrist)
    if elbow_angle > 80 and elbow_angle < 120:
        label = 'Pushup'
        color = (0, 255, 0)
        beep_sound.play()
        previous_label = 'Pushup'

    return label, color

pygame.mixer.init()
beep_sound = pygame.mixer.Sound('beep.wav')

def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle

def gen_frames():
    global current_frame
    video = cv2.VideoCapture(0)
    while True:
        success, frame = video.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ = frame.shape
            frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
            
            with frame_lock:
                current_frame = frame.copy()
                frame, landmarks = detectPose(frame, pose)
                label, color = classifyPose(landmarks)
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

                # Draw lines for gym curls and pushups
                if label in ['Biceps Curl', 'Pushup']:
                    if label == 'Biceps Curl':
                        cv2.line(frame, (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][0], landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][1]),
                                 (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value][0], landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value][1]), (0, 255, 0), 2)
                        cv2.line(frame, (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value][0], landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value][1]),
                                 (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value][0], landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value][1]), (0, 255, 0), 2)

                    if label == 'Pushup':
                        cv2.line(frame, (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][0], landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][1]),
                                 (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value][0], landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value][1]), (0, 0, 255), 2)
                        cv2.line(frame, (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value][0], landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value][1]),
                                 (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value][0], landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value][1]), (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def update_pose_status():
    global pose_status
    while True:
        with frame_lock:
            if current_frame is not None:
                frame, landmarks = detectPose(current_frame, pose)
                label, _ = classifyPose(landmarks)
                pose_status = label
        # Sleep to reduce CPU usage
        time.sleep(1)

@app.route('/status')
def pose_status_updates():
    def generate():
        while True:
            yield f"data: {pose_status}\n\n"
            time.sleep(1)  # Update the status every second
    return Response(generate(), content_type='text/event-stream')

@app.route('/curl_count')
def curl_count_updates():
    def generate():
        while True:
            yield f"data: {curl_count}\n\n"
            time.sleep(1)  # Update the curl count every second
    return Response(generate(), content_type='text/event-stream')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pose_detection')
def pose_detection():
    return render_template('pose_detection.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/dietian')
def diet_chart():
    return render_template('diet_chart.html')

@app.route('/profile_management')
@login_required
def profile_management():
    return render_template('profile_management.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/profile')
@login_required
def profile():
    user_progress = Progress.query.filter_by(user_id=current_user.id).all()
    achievements = Achievement.query.filter_by(user_id=current_user.id).all()
    return render_template('profile.html', progress=user_progress, achievements=achievements)

@app.route('/add_progress', methods=['POST'])
@login_required
def add_progress():
    exercise_name = request.form.get('exercise_name')
    count = int(request.form.get('count'))
    progress = Progress(exercise_name=exercise_name, count=count, user_id=current_user.id)
    db.session.add(progress)
    db.session.commit()
    return redirect(url_for('profile'))

@app.route('/achievements')
@login_required
def achievements():
    user_achievements = Achievement.query.filter_by(user_id=current_user.id).all()
    return render_template('achievements.html', achievements=user_achievements)

@app.route('/challenges')
@login_required
def challenges():
    challenges = Challenge.query.all()
    return render_template('challenges.html', challenges=challenges)

@app.route('/complete_challenge/<int:challenge_id>', methods=['POST'])
@login_required
def complete_challenge(challenge_id):
    challenge = Challenge.query.get(challenge_id)
    if challenge:
        progress = ChallengeProgress.query.filter_by(challenge_id=challenge_id, user_id=current_user.id).first()
        if not progress:
            progress = ChallengeProgress(challenge_id=challenge_id, user_id=current_user.id, progress=100)
            db.session.add(progress)
            db.session.commit()
            achievement = Achievement(name=f"Completed {challenge.name}", description=f"You completed the {challenge.name} challenge!", user_id=current_user.id)
            db.session.add(achievement)
            db.session.commit()
    return redirect(url_for('achievements'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:  # Note: In a real application, use hashed passwords
            login_user(user)
            return redirect(url_for('index'))
        return 'Invalid username or password'
    return render_template('profile_management.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

if __name__ == '__main__':
    threading.Thread(target=update_pose_status, daemon=True).start()
    app.run(debug=True)
