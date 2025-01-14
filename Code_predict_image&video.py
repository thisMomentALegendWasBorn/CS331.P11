import cv2
from ultralytics import YOLO
import time

# Define the output video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('your_output_path', fourcc, 20.0, (2000, 1000))

model = YOLO('your_model_yolo11n.pt')
input_path = 'your_input_path'  # Hoặc video path
model.to('cuda')

output_width = 2000
output_height = 1000

if input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
    # Đọc ảnh
    frame = cv2.imread(input_path)

    start_time = time.time()
    results = model(frame, conf=0.4)
    end_time = time.time()
    inference_time = end_time - start_time

    print(f"Thời gian suy luận: {inference_time:.4f} giây")

    annotated_frame = results[0].plot()

    cv2.imshow('YOLO Detection', annotated_frame)
    cv2.waitKey(0)

else:
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Không thể mở video.")
        exit()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    prev_frame_time = 0
    fps_list = []
    inference_times = []

    while(True):
        ret, frame = cap.read()

        if not ret:
            print("Hết video.")
            break

        start_time = time.time()
        # Áp dụng conf=0.4 vào đây
        results = model(frame, conf=0.4)
        end_time = time.time()
        inference_time = end_time - start_time
        inference_times.append(inference_time)

        annotated_frame = results[0].plot()

        resized_frame = cv2.resize(annotated_frame, (output_width, output_height))

        # Tính FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps_list.append(int(fps))

        # Hiển thị FPS
        cv2.putText(resized_frame, f'FPS: {int(fps)}', (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)

        # Save the frame to the output video
        out.write(resized_frame)

        cv2.imshow('YOLO Detection', resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()

    avg_fps = sum(fps_list) / len(fps_list)
    print(f"FPS trung bình: {avg_fps:.2f}")

    avg_inference_time = sum(inference_times) / len(inference_times)
    print(f"Thời gian suy luận trung bình: {avg_inference_time:.4f} giây")

cv2.destroyAllWindows()