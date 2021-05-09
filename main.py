import mediapipe as mp
import os.path as ops
import numpy as np
import torch
import cv2
import time
import os
import matplotlib.pylab as plt
import sys
from tqdm import tqdm
import imageio
from google.colab.patches import cv2_imshow

# Define detectors
mp_pose = mp.solutions.pose
pose = mp.solutions.pose.Pose(
  static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
face_detection = mp.solutions.face_detection.FaceDetection(
    min_detection_confidence=0.5)


# Initialize result var
is_face_distract = False
is_hands_off_wheel = False


# Initialize result change counter
face_distract_counter = 0
hands_off_counter = 0

# face_distract_time
face_distract_time = 0
hands_off_time = 0

fps = 1


# Set drawing spec
mp_drawing = mp.solutions.drawing_utils 
## face_detection
face_keypoint_spec = mp_drawing.DrawingSpec(color=(255, 255, 255))
normal_spec = mp_drawing.DrawingSpec(color=(0, 255, 0))
warning_spec = mp_drawing.DrawingSpec(color=(0, 0, 255))


# Prepare DrawingSpec for drawing the face landmarks later.
mp_drawing = mp.solutions.drawing_utils 
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Define Debug Mode
DEBUG = False


def detect_hands_on_steer(image):
  global is_hands_off_wheel, hands_off_counter, hands_off_time
  # Convert the BGR image to RGB and process it with MediaPipe Pose.
  results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

  # Print nose landmark.
  image_height, image_width, _ = image.shape
  if not results.pose_landmarks:
    return

  # Get hands pinky and thumb position ratio
  left_pinky = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY]
  left_thumb = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB]
  left_hand_pos = {'x': (left_pinky.x + left_thumb.x) / 2,
                   'y': (left_pinky.y + left_thumb.y) / 2}

  right_pinky = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY]
  right_thumb = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY]
  right_hand_pos = {'x': (right_pinky.x + right_thumb.x) / 2,
                    'y': (right_pinky.y + right_thumb.y) / 2}

  # Check hands on wheel
  left_hand_on_wheel = False
  right_hand_on_wheel = False
  if (left_hand_pos['x'] < 0.3) and (0.5 < left_hand_pos['y'] < 0.75):
    left_hand_on_wheel = True
  if (right_hand_pos['x'] < 0.37) and (0.1 < right_hand_pos['y'] < 0.5):
    right_hand_on_wheel = True

  frame_both_hand_on_wheel = left_hand_on_wheel and right_hand_on_wheel

  # Update hands off counter
  if (not frame_both_hand_on_wheel and not is_hands_off_wheel) or (frame_both_hand_on_wheel and is_hands_off_wheel):
    hands_off_counter += 1
  else:
    hands_off_counter = 0

  # Update hands off flag
  if hands_off_counter > 15:
    hands_off_counter = 0
    is_hands_off_wheel = not is_hands_off_wheel
    if not is_hands_off_wheel:
      hands_off_time = 0


  if DEBUG:
    print(f"""
  left_hand_pos: {left_hand_pos}
  left_hand_on_wheel: {left_hand_on_wheel}

  right_hand_pos: {right_hand_pos}
  right_hand_on_wheel: {right_hand_on_wheel}
  ---------
  hands_off_counter: {hands_off_counter}
  is_hands_off_wheel: {is_hands_off_wheel}
    """)


  '''
  print(
    f'Nose coordinates: ('
    f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
    f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_hight})'
  )
  '''
  connection_drawing_spec = normal_spec if not is_hands_off_wheel else warning_spec

  # Draw pose landmarks.
  annotated_image = image.copy()
  mp_drawing.draw_landmarks(
      image=annotated_image,
      landmark_list=results.pose_landmarks,
      connections=mp_pose.POSE_CONNECTIONS,
      landmark_drawing_spec=drawing_spec,
      connection_drawing_spec=connection_drawing_spec)
  return annotated_image



def detect_face(image):
  global is_face_distract, face_distract_counter, face_distract_time
  # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
  results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

  # Draw face detections of each face.
  if not results.detections:
    return

  # Get a result of detection
  detection = results.detections.pop()

  # Extract main keypoints
  keypoint_list = detection.location_data.relative_keypoints
  left_eye = keypoint_list[0]
  right_eye = keypoint_list[1]
  left_ear = keypoint_list[4]
  right_ear = keypoint_list[5]

  # Calculate frame distraction
  diff = abs(abs(left_eye.x-left_ear.x) - abs(right_eye.x-right_ear.x))
  frame_distracted = diff > 0.05

  # Update face distract counter
  if (frame_distracted and not is_face_distract) or (not frame_distracted and is_face_distract):
    face_distract_counter += 1
  else:
    face_distract_counter = 0

  # Update face distract flag
  if face_distract_counter > 10:
    face_distract_counter = 0
    is_face_distract = not is_face_distract
    if not is_face_distract:
      face_distract_time = 0

  if DEBUG:
    print(f"""
  diff:                    {diff}
  frame_distracted:        {frame_distracted}
  --------
  face_distract_counter:   {face_distract_counter}
  face_distract:           {is_face_distract}
    """)

  annotated_image = image.copy()
  face_bbox_spec = normal_spec if not is_face_distract else warning_spec
  mp_drawing.draw_detection(annotated_image, detection, keypoint_drawing_spec=face_keypoint_spec, bbox_drawing_spec=face_bbox_spec)
  return annotated_image



def display_state(display_frame):
  global face_distract_time, hands_off_time
  # Define color codes
  RED = (0, 0, 255)
  GREEN = (0, 255, 0)
  WHITE = (255, 255, 255)
  ORANGE = (69, 146, 249)

  # Initalize
  FONT = cv2.FONT_HERSHEY_SIMPLEX
  LINE = cv2.LINE_AA

  # Set text position
  display_x_t = 50
  display_x_v = 80
  display_y = (120, 190, 290, 360, 460, 530)
  title_scale = 1.3
  value_scale = 1.8

  # Calculate values
  face_value_color = GREEN
  hands_value_color = GREEN
  risk_value_color = GREEN
  risk_text = 'LOW'

  if is_face_distract:
    face_distract_time += 1
  
  if is_hands_off_wheel:
    hands_off_time += 1

  face_distract_time_secs = face_distract_time / fps
  hands_off_time_secs = hands_off_time / fps

  # Change text color, risk value
  if hands_off_time_secs > 5:
    hands_value_color = ORANGE
    risk_value_color = ORANGE
    risk_text = 'MEDIUM'

  if face_distract_time_secs > 5:
    face_value_color = ORANGE
    risk_value_color = RED
    risk_text = 'HIGH'

  if face_distract_time_secs > 8:
    face_value_color = RED

  # Write text
  cv2.putText(display_frame, 'Off Road Glance:', (display_x_t, display_y[0]), FONT, title_scale, WHITE, 3, LINE)
  cv2.putText(display_frame, '%.1f secs' % face_distract_time_secs, (display_x_v, display_y[1]), FONT, value_scale, face_value_color, 3, LINE)
  cv2.putText(display_frame, 'Off Wheel Hand:', (display_x_t, display_y[2]), FONT, title_scale, WHITE, 3, LINE)
  cv2.putText(display_frame, '%.1f secs' % hands_off_time_secs, (display_x_v, display_y[3]), FONT, value_scale, hands_value_color, 3, LINE)
  cv2.putText(display_frame, 'Estimated Risk:', (display_x_t, display_y[4]), FONT, title_scale, WHITE, 3, LINE)
  cv2.putText(display_frame, risk_text, (display_x_v, display_y[5]), FONT, value_scale, risk_value_color, 3, LINE)

  return display_frame



def main(video_path):
    global fps
    # video to frames
    col_input_path = os.path.join(video_path,'a_column_driver.mp4')
    wheel_input_path = os.path.join(video_path,'steering_wheel.mp4')

    col_capture = cv2.VideoCapture(col_input_path)
    col_has_frame, col_frame = col_capture.read()
    wheel_capture = cv2.VideoCapture(wheel_input_path)
    wheel_has_frame, wheel_frame = wheel_capture.read()


    if not DEBUG:
        col_output_file = os.path.join(video_path,'output-a_column_driver.mp4')
        wheel_output_file = os.path.join(video_path,'output-steering_wheel.mp4')
        display_output_file = os.path.join(video_path,'output-display.mp4')
    frame_width = int(col_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(col_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = col_capture.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if not DEBUG:
        col_video = cv2.VideoWriter(col_output_file, fourcc, 25, (frame_width, frame_height))
        wheel_video = cv2.VideoWriter(wheel_output_file, fourcc, 25, (frame_width, frame_height))
        display_video = cv2.VideoWriter(display_output_file, fourcc, 25, (512, 640))


    # Extract lane from frame
    while True:
        col_has_frame, col_frame = col_capture.read()
        wheel_has_frame, wheel_frame = wheel_capture.read()

        if not col_has_frame or not wheel_has_frame:
            break
          
        display_frame = np.zeros((640, 512, 3), np.uint8)
        col_output_frame = detect_hands_on_steer(col_frame)
        wheel_output_frame = detect_face(wheel_frame)
        display_frame = display_state(display_frame)
        
        if not isinstance(col_output_frame, type(None)) and not isinstance(wheel_output_frame, type(None)):
            # frames to video & store video

            if DEBUG:
                cv2_imshow(output_frame)
            else:
                col_video.write(col_output_frame)
                wheel_video.write(wheel_output_frame)
                #display_video.write(display_frame)

    col_capture.release()   
    wheel_capture.release()   

    if not DEBUG: 
        col_video.release()
        wheel_video.release()
        display_video.release()



if __name__ == '__main__':
    video_path = "/content/gdrive/My Drive/Colab Notebooks/" # input your video path
    main(video_path)