# Air Touch
A virtual mouse that performs mouse operations using hand gestures

### To run the below mouse in your local system

`git clone https://github.com/Jay4Codes/AirTouch.git`

`pip install -r requirements.txt`

`python gesture_mouse.py`

## SVC
For every new user, we ask the user to record hand movements through their webcam.
Now, the model detects hand poses and generates a new dataset.
According to this new dataset, the Support Vector Classifier Model parameters can be retrained. SVC is lightweight so re-training for every user will be faster. and also provides good results for multi-class classification.

dataset_train.csv - Dataset of coordinates <br>
`delay.py` - Adds delays and time thresholding for smooth execution <br>
`generate_data.py` - To generate new data for each new user <br>
`hand_detect.py` - Detect hands using media pipe <br>
`hand_movements.py` - Mouse executions as per the gestures identified <br>
`hand_pose_transform.py` - Pose transformation <br>
`hand_poses` - Pose classifier <br>
`handsPoseClassifier.pkl` - pickle model file <br>
`train_hands_poses_classifier.py` - trains new model file on new datasets <br>
`utils.py` - basic utilities required <br>

## YOLO
For every new user, we ask the user to record hand movements through their webcam.
Now, the model will be re-trained on 5-10 snapshots for each pose and the YOLO parameters will be tuned respectively. 

`detect.py` - python executable file that detects hands and classifies gestures <br>
`hand.pt` - Our own trained gesture model that returns gestures identified <br>

## SSD
--

SVC Accuracy is quite less compared to YOLO but faster than YOLO.

