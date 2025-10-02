import os
import cv2


def split_video_to_frames(path, rot=-1, grayscale=False):
    video_name = path.split("\\")[-1].split(".")[0]
    cap = cv2.VideoCapture(path)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        i += 1
        print(i)
        if rot != -1:
            frame = cv2.rotate(frame, rot)
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        os.makedirs(fr'./Outputs/{video_name}/Frames/', exist_ok=True)
        cv2.imwrite(fr'./Outputs/{video_name}/Frames/Frame_{i}.jpg', frame)
    cap.release()


dir_path = r"C:\Physics\Year 2\Lab\Advanced Lab 2B\21.05\\"
file_names = os.listdir(dir_path)
for fn in file_names:
    if fn != "long_337 RPM.mp4":
        continue
    split_video_to_frames(dir_path + fn, grayscale=True)

# video_path = r"C:\Physics\Year 2\Lab\Advanced Lab 2B\21.05\long_189 RPM.mp4"
# split_video_to_frames(video_path, grayscale=True)

