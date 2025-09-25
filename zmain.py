from DPVSA.zmodel import HybridModel
import cv2

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    model = HybridModel(yoloPath="models/yolov8.pt")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        out = model.inferFrame(frame, frameSize=[w, h])
        seqOut = model.inferSequence()
        print("frameDetections:", [ (d.label, round(d.confidence,2)) for d in out["detections"] ], "seqLogitsShape:", seqOut.shape)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()