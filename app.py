from flask import Flask, request, render_template, redirect, session, url_for, Response
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.orm import sessionmaker
from model import engine, User
import cv2
import time

app = Flask(__name__, template_folder='templates')
app.secret_key = 'aptarsecretkey'
Session = sessionmaker(bind=engine)
db_session = Session()




@app.route('/')
def login():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def authenticate():
    username = request.form['username']
    password = request.form['password']
    user = db_session.query(User).filter_by(username=username).first()

    if user:
        print(f"User found: {user.username}, Role: {user.role}")  # Debugging

    if user and check_password_hash(user.password, password):
        session['user'] = user.username
        session['role'] = user.role
        session.modified = True  # Ensures session update
        print("Login successful - Redirecting to dashboard")  # Debugging
        return redirect(url_for('dashboard'))
    
    print("Invalid login attempt!")  # Debugging
    return "Invalid credentials!", 401

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', user=session['user'])

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    print(f"Session data: {session}")  # Debugging
    if 'user' not in session or session.get('role', '') != 'admin':
        print("Unauthorized access - Redirecting to login")  # Debugging
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        username = request.form['username']
        existing_user = db_session.query(User).filter_by(username=username).first()
        
        if existing_user:
            print(f"User {username} already exists!")  # Debugging
            return "User already exists!", 400  # Send a proper error message
        
        password = generate_password_hash(request.form['password'])
        role = request.form['role']
        new_user = User(username=username, password=password, role=role)
        db_session.add(new_user)
        db_session.commit()
    
    return render_template('admin.html')

# Define video sources
VIDEO_SOURCES = {
    "camera1": "./static/videos/Double stacked_warehouse_business_pallets_work.mp4",
    "camera2": "./static/videos/Order_selector_really_fast_paced_warehouse.mp4",
    "cctv_stream": "./static/videos/Order_selector_really_fast_paced_warehouse.mp4" 
}

def generate_frames(video_path):
    """Reads frames from a video file and streams them at a slower speed."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the original FPS of the video
    frame_delay = 1 / fps if fps > 0 else 0.05  # Avoid division by zero

    while True:
        success, frame = cap.read()
        if not success:
            print(f"Restarting video: {video_path}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(frame_delay * 1.5)  # Increase this factor to slow down the video more

    cap.release()

@app.route('/video_feed/<camera>')
def video_feed(camera):
    """Serves the selected video stream to the frontend."""
    if camera in VIDEO_SOURCES:
        return Response(generate_frames(VIDEO_SOURCES[camera]),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    return "Camera Not Found", 404



@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('role', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')