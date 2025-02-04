
# Pose Detection Web App

Welcome to the **Pose Detection Web App** built using Python, OpenCV, MediaPipe, and Flask! This app allows users to:

- Capture custom poses from a live video stream.
- Detect poses using the webcam and match them to stored custom poses.
- Save and delete poses in the system.

## Features

- **Pose Detection**: Real-time pose detection using MediaPipe Pose.
- **Custom Pose Storage**: Capture and store custom poses with joint angles.
- **Live Video Feed**: View your live webcam stream with pose overlays.
- **Web Interface**: User-friendly web interface for adding, deleting, and detecting poses.

## How It Works

1. **Add Pose**: Capture a frame from the webcam and save the pose with a name.
2. **Delete Pose**: Remove a previously saved custom pose.
3. **Detect Pose**: Use live webcam feed to detect and match poses in real-time.

## Installation

### Requirements

- Python 3.7+
- Flask
- OpenCV
- Mediapipe
- JSON (for storing custom poses)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/GitHub-JeetBhardwaj/PoseEstimator.git
   ```

2. Go to that directory
   
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask app:
   ```bash
   python app.py
   ```

5. Open your browser and go to `http://127.0.0.1:5000/` to access the web interface.

## Web Interface

The app has the following web pages:

- **Home Page**: Introduction and access to the app features.
- **Add Pose**: Capture and save custom poses.
- **Delete Pose**: Remove custom poses.
- **Detect Pose**: Use live video feed to detect and match poses.

## Preview

Here are some screenshots of the app in action:

![Preview 1](https://github.com/GitHub-JeetBhardwaj/PoseEstimator/blob/main/Assets/p1.png)
![Preview 2](https://github.com/GitHub-JeetBhardwaj/PoseEstimator/blob/main/Assets/p2.png)
![Preview 3](https://github.com/GitHub-JeetBhardwaj/PoseEstimator/blob/main/Assets/p3.png)
![Preview 4](https://github.com/GitHub-JeetBhardwaj/PoseEstimator/blob/main/Assets/p4.png)
![Preview 5](https://github.com/GitHub-JeetBhardwaj/PoseEstimator/blob/main/Assets/p6.png)
![Preview 6](https://github.com/GitHub-JeetBhardwaj/PoseEstimator/blob/main/Assets/p7.png)
![Preview 7](https://github.com/GitHub-JeetBhardwaj/PoseEstimator/blob/main/Assets/p8.png)
![Preview 7](https://github.com/GitHub-JeetBhardwaj/PoseEstimator/blob/main/Assets/p9.png)
![Preview 7](https://github.com/GitHub-JeetBhardwaj/PoseEstimator/blob/main/Assets/p10.png)

## Technologies Used

- **Flask**: Web framework for the backend.
- **OpenCV**: Real-time computer vision for video capture and processing.
- **MediaPipe**: Pose detection library.
- **JSON**: For storing custom poses.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

If you have any questions or need further assistance, feel free to open an issue or reach out!

**Creator:** Jeet Bhardwaj
