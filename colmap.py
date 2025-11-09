import cv2, os
os.makedirs('images', exist_ok=True)
cap = cv2.VideoCapture('ComputerVisionVideo.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
interval = int(fps // 2)  # 1 frame every 0.5 seconds, tweak as needed
count, img_idx = 0, 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if count % interval == 0:
        cv2.imwrite(f'images/frame_{img_idx:03d}.jpg', frame)
        img_idx += 1
    count += 1
cap.release()
print(f"Extracted {img_idx} frames.")
