from ultralytics import YOLO
import cv2
from moviepy.editor import VideoFileClip
import os


model = YOLO("model/best.pt")
names = model.names

video_folder = "videos"

output_folder = "audiovideos_cropped"

os.makedirs(output_folder, exist_ok=True)

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    assert cap.isOpened(), "error"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    overlay_img = cv2.imread('logoimage.png', cv2.IMREAD_UNCHANGED)

    video_writer = cv2.VideoWriter(output_path,
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps, (w, h), isColor=True)

    prev_boxes = []

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("success.")
            break

        results = model.predict(im0, show=False)
        boxes = results[0].boxes.xyxy.cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()

        if boxes:
            prev_boxes = boxes
            for box, cls in zip(boxes, clss):
                obj = im0[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                blur_obj = cv2.medianBlur(obj, 249) 
                im0[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = blur_obj

                detected_width = int(box[2]) - int(box[0])
                detected_height = int(box[3]) - int(box[1])

                overlay_resized = cv2.resize(overlay_img, (detected_width, detected_height))

                if overlay_resized.shape[2] == 4:
                    overlay_rgb = overlay_resized[:, :, :3]
                    alpha = overlay_resized[:, :, 3:] / 255.0
                    im0[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = (
                            alpha * overlay_rgb + (1 - alpha) * im0[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    )
                else:
                    im0[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = overlay_resized
        elif prev_boxes:
            for box in prev_boxes:
                expanded_box = [
                    max(0, int(box[0]) - 10),  
                    max(0, int(box[1]) - 10),  
                    min(w, int(box[2]) + 10),  
                    min(h, int(box[3]) + 10)   
                ]

                obj = im0[expanded_box[1]:expanded_box[3], expanded_box[0]:expanded_box[2]]
                blur_obj = cv2.medianBlur(obj, 299) 
                im0[expanded_box[1]:expanded_box[3], expanded_box[0]:expanded_box[2]] = blur_obj

                detected_width = expanded_box[2] - expanded_box[0]
                detected_height = expanded_box[3] - expanded_box[1]

                overlay_resized = cv2.resize(overlay_img, (detected_width, detected_height))

                if overlay_resized.shape[2] == 4:
                    overlay_rgb = overlay_resized[:, :, :3]
                    alpha = overlay_resized[:, :, 3:] / 255.0
                    im0[expanded_box[1]:expanded_box[3], expanded_box[0]:expanded_box[2]] = (
                            alpha * overlay_rgb + (1 - alpha) * im0[expanded_box[1]:expanded_box[3], expanded_box[0]:expanded_box[2]]
                    )
                else:
                    im0[expanded_box[1]:expanded_box[3], expanded_box[0]:expanded_box[2]] = overlay_resized

        video_writer.write(im0)

    cap.release()
    video_writer.release()

    video_clip = VideoFileClip(output_path)
    audio_clip = VideoFileClip(input_path).audio 
    video_clip = video_clip.set_audio(audio_clip)

    cropped_clip = video_clip.crop(x1=50, y1=50, x2=w-50, y2=h-50)

    output_cropped_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_path))[0]}_cropped.mp4")
    cropped_clip.write_videofile(output_cropped_path, codec="libx264", audio_codec="aac")

    video_clip.close()
    cropped_clip.close()

for video_filename in os.listdir(video_folder):
    if video_filename.endswith(".mp4"):
        video_path = os.path.join(video_folder, video_filename)
        output_path = os.path.join(output_folder, f"{video_filename}_processed.mp4")
        process_video(video_path, output_path)

        processed_path = os.path.join(output_folder, f"{video_filename}_processed.mp4")
        if os.path.exists(processed_path):
            os.remove(processed_path)
