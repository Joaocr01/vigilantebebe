# api/index.py
from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO

app = Flask(__name__)

modelo = YOLO('best.pt')
video = cv2.VideoCapture('ex3.mp4')
area = [430, 240, 930, 280]
paused = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/toggle_pause')
def toggle_pause():
    global paused
    paused = not paused
    return jsonify({"status": "paused" if paused else "playing"})

def gerar_frames():
    global paused
    while True:
        if paused:
            continue

        check, img = video.read()
        if not check:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        img = cv2.resize(img, (1270, 720))
        img2 = img.copy()
        cv2.rectangle(img2, (area[0], area[1]), (area[2], area[3]), (0, 255, 0), -1)
        resultados = modelo(img)
        bebeDentroDaArea = False

        for resultado in resultados:
            boxes = resultado.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls = int(box.cls[0].item())
                label = modelo.names[cls]

                if label == 'bebe':
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 5)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    if area[0] <= cx <= area[2] and area[1] <= cy <= area[3]:
                        bebeDentroDaArea = True

        if bebeDentroDaArea:
            cv2.rectangle(img2, (area[0], area[1]), (area[2], area[3]), (0, 0, 255), -1)
            cv2.rectangle(img, (100, 30), (470, 80), (0, 0, 255), -1)
            cv2.putText(img, 'BEBE EM PERIGO', (105, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        imgFinal = cv2.addWeighted(img2, 0.5, img, 0.5, 0)
        ret, jpeg = cv2.imencode('.jpg', imgFinal)
        if not ret:
            continue

        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gerar_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
