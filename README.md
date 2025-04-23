# 🚗 Blurshield: Real-Time Vehicle Tracking & Blurring

**Blurshield** is a privacy-preserving, real-time vehicle tracking system that integrates YOLOv8 for object detection with Deep SORT for multi-object tracking. It provides dynamic Gaussian blurring of vehicles to anonymize video surveillance while preserving situational awareness. The system is built with a **Flask backend** and a **MERN stack frontend**, optimized for deployment in sensitive surveillance contexts such as military, traffic control, or smart cities.

---

## 📦 Features

- 🔍 **Real-Time YOLOv8 Detection** — Accurate detection of vehicles under varied lighting and weather.
- 🛰 **Deep SORT-Based Tracking** — Multi-object tracking using re-identification for persistent IDs.
- 🛡 **Dynamic Vehicle Blurring** — Applies Gaussian blur to anonymize detected vehicles.
- 🖥 **Flask API Backend** — Efficient backend pipeline for real-time frame processing.
- 🌐 **Edge-Ready Deployment** — Designed to support GPU acceleration for high-throughput environments.

---

## 🧰 Technologies Used

- **YOLOv8**: Object detection
- **Deep SORT (Ziqiang, MIT License)**: Object tracking
- **Flask**: RESTful API and backend processing
- **OpenCV + Torch**: Image and tensor operations

---

## 🚀 Getting Started

Follow the steps below to get your Blurshield system up and running.

**Prerequisites**
- Python 3.8+

- GPU support (Optional, but recommended for real-time performance)

Installation Steps
1. **Clone the repository:**
    git clone `https://github.com/yourusername/blurshield.git`
    `cd blurshield`

2. **Set up the Flask Backend:**
   - **Create a virtual environment for Python:**
        `python -m venv venv`
   
   - **Activate the virtual environment:**
        - For Windows:
          `venv\Scripts\activate`
        - For Linux/macOS:
            `source venv/bin/activate`
     
   - **Install the required dependencies:**
        `pip install -r requirements.txt`
   
3. **Configure YOLO Weights:**
    - Download the YOLOv8 pre-trained weights (yolov8n.pt, yolov8m.pt, yolov8s.pt) and place
      them in the YOLO-Weights/ directory.
    - For the Deep SORT model, ensure the deep_sort_pytorch directory has all required files.
   
4. **Run the Flask App:**
    - In the backend directory, start the Flask server:
        `cd FlaskAppBlurshield-YOLOv8`
        `python flaskapp.py`
   

## 🌐 API Documentation
The Flask API serves real-time vehicle tracking and blurring requests. Below are the key API endpoints:

**POST** `/process-video`
- **Description:** Process a video for vehicle tracking and blurring.

- **Parameters:**

  - `video`: Video file (MP4, AVI, etc.).

- **Response:**

  - `status`: `success` or `failure`.

  - `blurred_video_url`: URL to the processed video.

**GET** `/status`
- **Description:** Get the current status of the video processing.

- **Response:**

    - `status`: Current processing status (e.g., `processing`, `completed`, `failed`).


## 🔒 Security Considerations
- **Authentication:** Add token-based authentication (e.g., JWT) for API access.

- **HTTPS:** Configure SSL certificates to ensure secure communication.


## ⚙️ Deployment Instructions
**To deploy this system on a cloud platform (e.g., AWS EC2 with GPU support):**

1. Set up your cloud server with Ubuntu 20.04 and NVIDIA GPU (e.g., T4).

2. Install required dependencies (CUDA, cuDNN, Python libraries).

3. Deploy Flask Backend using Gunicorn or uWSGI, and set up Nginx as a reverse proxy.


## 📄 License
**This project uses third-party open source components. Notably:**

- Deep SORT by [Ziqiang](https://github.com/ZQPei/deep_sort_pytorch)
 is used for multi-object tracking.
Licensed under the [MIT License](https://github.com/ZQPei/deep_sort_pytorch/blob/master/LICENSE).

    

## 📁 Folder Structure

Blurshield/
├── FlaskAppBlurshield-YOLOv8/
│   ├── deep_sort_pytorch/
│   ├── static/
│       ├── files
│       ├── images
│   ├── templates/
│       ├── index.html
│       ├── ui.html
│       ├── videoprojectnew.html
│   ├── bus.jpg
│   ├── flaskapp.py
│   └── YOLO_Video.py
│   ├── yolov8n.pt
├── THIRD_PARTY_LICENSES/
│   ├── deep_sort_LICENSE/
│       ├── LICENSE
├── venv
├── YOLO-Weights/
│   ├── yolov8m.pt
│   ├── yolov8n.pt
│   ├── yolov8s.pt
├── LICENSE
├── README.md
├── requirements.txt


## Acknowledgments
We would like to express our sincere gratitude to **Dr. V.S. Wadne** for their invaluable guidance and support throughout the development of this project. Dr. Wadne's expertise and insightful feedback played a key role in shaping the direction and success of the **Blurshield: Real-Time Vehicle Tracking and Blurring System**.

## Contributors

- **Aman Rajjak Sayyad:** Lead Developer, YOLO Model and Object Tracking
- **Amit Kumbhar:** Front-end Development, Web App Interface
- **Aman Rajjak Sayyad:** Backend Development, Deployment
- **Dr. V.S Wadne:** Guidance

Special thanks to all team members for their contributions to the project.
