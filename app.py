import cv2
import mediapipe as mp
import math
import json
import os
from flask import Flask, render_template, request, redirect, url_for

# Initialize Flask app
app = Flask(__name__)

# Initialize Mediapipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# File to store custom poses
POSES_FILE = "custom_poses.json"

# Load custom poses from the file if it exists
def load_custom_poses():
    if os.path.exists(POSES_FILE):
        try:
            with open(POSES_FILE, "r") as file:
                return json.load(file)
        except json.JSONDecodeError:
            print(f"Error: The file {POSES_FILE} is empty or contains invalid JSON.")
            return {}
    return {}

# Save custom poses to the file
def save_custom_poses():
    with open(POSES_FILE, "w") as file:
        json.dump(custom_poses, file)

# Store custom poses with their joint angles (in memory)
custom_poses = load_custom_poses()

def calculate_angle(a, b, c):
    """
    Calculate the angle (in degrees) between three points: a, b, c.
    a, b, c are tuples of (x, y).
    """
    # Calculate vectors
    ba = (a[0] - b[0], a[1] - b[1])  # Vector BA
    bc = (c[0] - b[0], c[1] - b[1])  # Vector BC

    # Dot product and magnitudes
    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    magnitude_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    magnitude_bc = math.sqrt(bc[0]**2 + bc[1]**2)

    # Handle edge cases for division by zero
    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0  # If either vector has zero magnitude, return 0 degrees

    # Calculate cosine of the angle and clamp it to the valid range [-1, 1]
    cos_angle = dot_product / (magnitude_ba * magnitude_bc)
    cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to [-1, 1]

    # Calculate and return angle in degrees
    angle = math.acos(cos_angle)
    return math.degrees(angle)


def extract_joint_angles(landmarks, h, w):
    """
    Extract angles between key joints from pose landmarks.
    """
    keypoints = [(int(landmark.x * w), int(landmark.y * h)) for landmark in landmarks]
    
    # Define key angles (shoulder, elbow, knee, hip, neck, etc.)
    angles = {
        "left_elbow": calculate_angle(keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      keypoints[mp_pose.PoseLandmark.LEFT_WRIST.value]),
        "right_elbow": calculate_angle(keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       keypoints[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       keypoints[mp_pose.PoseLandmark.RIGHT_WRIST.value]),
        "left_knee": calculate_angle(keypoints[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     keypoints[mp_pose.PoseLandmark.LEFT_ANKLE.value]),
        "right_knee": calculate_angle(keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      keypoints[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      keypoints[mp_pose.PoseLandmark.RIGHT_ANKLE.value]),
        "neck_angle": calculate_angle(keypoints[mp_pose.PoseLandmark.NOSE.value],
                                      keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value]),  # Neck angle with shoulder
        "left_hip_angle": calculate_angle(keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                          keypoints[mp_pose.PoseLandmark.LEFT_HIP.value],
                                          keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value]),
        "right_hip_angle": calculate_angle(keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                           keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                           keypoints[mp_pose.PoseLandmark.RIGHT_KNEE.value]),
    }
    return angles

def match_pose(current_angles):
    """
    Match the current pose against saved poses using a tolerance.
    """
    if not custom_poses:
        return None

    tolerance = 20  # Allow a 20-degree error margin
    for pose_name, saved_angles in custom_poses.items():
        match = all(
            abs(current_angles[joint] - saved_angles[joint]) <= tolerance
            for joint in saved_angles
        )
        if match:
            return pose_name
    return None

def add_custom_pose_from_live_frame(pose_name):
    """
    Capture a frame from live video feed, show the user the frame, and prompt for pose name.
    """
    cap = cv2.VideoCapture(0)
    print("Press 'c' to capture the current pose and enter a name.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)

        if result.pose_landmarks:
            # Draw pose landmarks on the frame
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the live frame
        cv2.imshow("Pose Detection - Press 'c' to capture", frame)
        cv2.setWindowProperty("Pose Detection - Press 'c' to capture",cv2.WND_PROP_TOPMOST,1)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') or key==ord('C'):  # When 'c' is pressed, capture the pose
            # Extract joint angles
            current_angles = extract_joint_angles(result.pose_landmarks.landmark, h, w)

            # Save the pose with the provided name
            custom_poses[pose_name] = current_angles
            save_custom_poses()
            print(f"Pose '{pose_name}' added successfully.")
            break  # Exit after adding the pose

        elif key == ord('q') or key==ord('Q'):  # Press 'q' to quit the program
            break

    cap.release()
    cv2.destroyAllWindows()

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html', name="JEET BHARDWAJ")

@app.route('/contacts')
def contacts():
    return render_template('contacts.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/add_pose', methods=['GET', 'POST'])
def add_pose():
    if request.method == 'POST':
        pose_name = request.form['pose_name']
        add_custom_pose_from_live_frame(pose_name)
    return render_template('add_pose.html')

@app.route('/delete_pose', methods=['GET', 'POST'])
def delete_pose():
    if request.method == 'POST':
        pose_name = request.form['pose_name']
        if pose_name in custom_poses:
            del custom_poses[pose_name]
            save_custom_poses()
    return render_template('delete_pose.html', poses=custom_poses)

@app.route('/detect_pose', methods=['GET'])
def detect_pose():
    cap = cv2.VideoCapture(0)
    pose_name = None  # Variable to store the pose name

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)

        if result.pose_landmarks:
            # Draw pose landmarks on the frame
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            current_angles = extract_joint_angles(result.pose_landmarks.landmark, h, w)
            pose_name = match_pose(current_angles)  # Find the matched pose

            if pose_name:
                # Display pose name on the screen
                cv2.putText(frame, f"Pose: {pose_name}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show live video with pose detection
        cv2.imshow("Pose Detection", frame)
        cv2.setWindowProperty("Pose Detection",cv2.WND_PROP_TOPMOST,1)
        # Check if the user clicked the cross (closed the window)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or cv2.getWindowProperty("Pose Detection", cv2.WND_PROP_VISIBLE) < 1:
            break  # Exit the loop if 'q' is pressed or the window is closed

    cap.release()
    cv2.destroyAllWindows()

    return render_template('detect_pose.html', pose_name=pose_name)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
