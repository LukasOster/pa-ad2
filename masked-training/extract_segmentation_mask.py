import cv2
import supervision as sv
from rfdetr import RFDETRSegPreview
from rfdetr.util.coco_classes import COCO_CLASSES
import os
import numpy as np

model = RFDETRSegPreview(pretrained=True, pretrain_weights=r"A:\Machine Learning\models\pa-model-library\pa-models-od3\out\pa_seamseg\checkpoint_best_total.pth")
video_path = r"A:\Machine Learning\datasets\raw-image-sets\bnzl\jd\20260505_112034.mp4"
cap = cv2.VideoCapture(video_path)


box_annotator  = sv.BoxAnnotator(thickness=2)
mask_annotator = sv.MaskAnnotator(opacity=0.4)  # <- draws filled masks
class_names=["plate"]

# ---- options ----
SAVE_SEGMENTED = True            # master switch
SAVE_DIR = video_path.split(".")[0]+"_masks"
SAVE_EACH_INSTANCE = False       # True: one image per mask; False: one combined image per frame
MIN_AREA_PX = 200                # filter tiny masks (optional)
# -----------------

os.makedirs(SAVE_DIR, exist_ok=True)

frame_idx = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    detections = model.predict(frame, device="cuda", return_masks=True, threshold=0.75)

    # labels for boxes (optional)
    labels = [
        f"{class_names[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    # 1) draw masks first (so boxes/text sit on top)
    frame_vis = mask_annotator.annotate(scene=frame.copy(), detections=detections)
    # 2) then draw boxes + labels
    frame_vis = box_annotator.annotate(scene=frame_vis, detections=detections)
    frame_vis = sv.LabelAnnotator().annotate(frame_vis, detections, labels)

    # ---- save "masked area on white background" ----
    if SAVE_SEGMENTED and getattr(detections, "mask", None) is not None and len(detections) > 0:
        # detections.mask is usually (N, H, W) boolean
        masks = detections.mask

        if SAVE_EACH_INSTANCE:
            for i, m in enumerate(masks):
                m = m.astype(bool)

                # optional: skip tiny masks
                if m.sum() < MIN_AREA_PX:
                    continue

                white_bg = np.full_like(frame, 255)       # white image
                white_bg[m] = frame[m]                    # paste original pixels inside mask

                cls = int(detections.class_id[i])
                conf = float(detections.confidence[i])
                out_path = os.path.join(
                    SAVE_DIR, f"frame_{frame_idx:06d}_det_{i:02d}_{class_names[cls]}_{conf:.2f}.png"
                )
                abs_path = os.path.abspath(out_path)

                ok = cv2.imwrite(abs_path, white_bg)
                print("WRITE:", ok, abs_path)

                print("EXISTS:", os.path.exists(abs_path))
                if os.path.exists(abs_path):
                    st = os.stat(abs_path)
                    print("SIZE:", st.st_size, "bytes")
                print(out_path)

                print("Image saved")

        else:
            # combined: union of all masks into one output
            union = np.any(masks.astype(bool), axis=0)

            if union.sum() >= MIN_AREA_PX:
                white_bg = np.full_like(frame, 255)
                white_bg[union] = frame[union]
                out_path = os.path.join(SAVE_DIR, f"frame_{frame_idx:06d}.png")
                abs_path = os.path.abspath(out_path)

                ok = cv2.imwrite(abs_path, white_bg)
                print("WRITE:", ok, abs_path)

                print("EXISTS:", os.path.exists(abs_path))
                if os.path.exists(abs_path):
                    st = os.stat(abs_path)
                    print("SIZE:", st.st_size, "bytes")
                print("Image saved")
    # -----------------------------------------------

    cv2.imshow("seg", frame_vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print(frame_idx)
    frame_idx += 1

cap.release()
cv2.destroyAllWindows()


#C:\Users\Lukas Oster\Nextcloud\Aussendarstellung\PA Bilder und Logos\PA WAAM.mp4