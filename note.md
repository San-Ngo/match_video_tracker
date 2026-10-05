
## Week 1 — Terminal, Virtual Environments & pytest

**What I learned:**
I learned how to use the terminal to create and manage my project, set up a virtual environment, and install the libraries I need. I also learned how pytest checks my code and how to read an error message to find the file and line where something went wrong.

## Week 2 — NumPy for Images

**What I learned:**
I learned that an image is basically a NumPy array, so I can use slicing to crop players and masks to select specific pixels. I also learned why images use height, width, and color channels, and why converting `uint8` to `float32` can prevent incorrect color calculations.

## Week 3 — OpenCV Basics

**What I learned:**
I learned how OpenCV reads images and videos frame by frame, converts between color spaces, and draws boxes, labels, and other shapes. I also learned how HSV can be used to create a mask that separates the football pitch from the rest of the frame.

## Week 4 — Ball & Possession

**What I learned:**
I learned that detecting the ball is harder than detecting players because small objects can disappear between frames. I also learned that YOLO sometimes detected spare balls near the boards, so I had to handle those false detections before calculating possession.

## Week 5 — Camera Motion

**What I learned:**
I learned that player movement in the video can be caused by both the players and the moving camera. The goal was to estimate the camera motion so I could separate camera movement from actual player movement.

## Week 6 — Spotlight

**What I learned:**
I learned how to highlight a tracked player by drawing a ring and a movement trace underneath them. Drawing these elements underneath the players makes the visualization clearer and keeps the players visible.
