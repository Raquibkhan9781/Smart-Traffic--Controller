AI SMART TRAFFIC CONTROLLER - CLEAN MVP

1) Folder structure
   Smart_Traffic_Final/
   |-- smart_traffic.py
   |-- requirements.txt
   |-- models/
   |   `-- yolo11n.pt
   `-- video/
       `-- traffic.mp4

2) Put your working yolo11n.pt inside models/.
3) Put the traffic MP4 inside video/ and name it traffic.mp4.
4) Open PowerShell in this folder.
5) Run:
      pip install -r requirements.txt
      python smart_traffic.py
6) Press Q to close.

WHAT THIS MVP DOES
- Reads a traffic video or camera.
- Detects standard YOLO vehicle classes.
- Tracks unique vehicle IDs using ByteTrack.
- Uses four calibrated approach gates for the uploaded overhead video.
- Classifies completed trips into LEFT / STRAIGHT / RIGHT using entry+exit.
- Counts each completed vehicle once.
- Calculates a prototype safe phase and green time using vehicle count x 2 seconds.

IMPORTANT
The uploaded test clip is short, so some vehicles may not complete an entry-to-exit trip.
Auto-rickshaw, e-rickshaw and tractor need a custom Indian-traffic model for reliable detection;
standard yolo11n.pt is not trained specifically for those classes.
For a real roadside controller, the safety/phase logic must be validated by traffic-safety engineers.
