import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

video_path = '01.mp4'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('%s_output.mp4' % (video_path.split('.')[0]), fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=2
)

is_first = True

first_center_x, first_center_y, first_radius = None, None, None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_h, img_w, _ = frame.shape

    img_result = frame.copy()


    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    mp_drawing.draw_landmarks(
        img_result,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )

    if results.pose_landmarks:
        landmark = results.pose_landmarks.landmark

        left_ear_x = landmark[mp_pose.PoseLandmark.LEFT_EAR].x * img_w
        left_ear_y = landmark[mp_pose.PoseLandmark.LEFT_EAR].y * img_h

        right_ear_x = landmark[mp_pose.PoseLandmark.RIGHT_EAR].x * img_w
        right_ear_y = landmark[mp_pose.PoseLandmark.RIGHT_EAR].y * img_h

        center_x = int((left_ear_x + right_ear_x) / 2)
        center_y = int((left_ear_y + right_ear_y) / 2)

        radius = int((left_ear_x - right_ear_x) / 2)
        radius = max(radius, 20)

        if is_first:
            first_center_x = center_x
            first_center_y = center_y
            first_radius = int(radius * 2)

            is_first = False
        else:
            cv2.circle(img_result, (first_center_x, first_center_y), radius=first_radius, color=(0, 0, 255), thickness=2)

            color = (0, 255, 0)

            if center_x - radius < first_center_x - first_radius \
                or center_x + radius > first_center_x + first_radius:
                color = (0, 0, 255)

            cv2.circle(img_result, (center_x, center_y), radius=radius, color=color, thickness=2)

        

    cv2.imshow('frame', img_result)
    out.write(img_result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pose.close()
cap.release()
out.release()