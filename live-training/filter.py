import cv2
from ultralytics import YOLO
import numpy as np
import torch
import supervision as sv
from PIL import Image
from rfdetr import RFDETRSmall

print("The operating system is Linux.")
#model = YOLO(r"/media/lukas-oster/C86C6AAC6C6A954A/Machine Learning/ml-tools-main/runs/detect/train13/weights/best.pt")
#model2 = YOLO(r"/media/lukas-oster/C86C6AAC6C6A954A/Machine Learning/ml-tools-main/runs/segment/train22-weld-seams4_yolov8seg/weights/best.pt")
model2 = YOLO(r"/home/pamv3/Desktop/models/seamseg2-measurement/weights/weld-seams7.pt")
model3 = RFDETRSmall(pretrained=True,pretrain_weights=r"/home/pamv3/Desktop/models/seamseg2-measurement/weights/checkpoint_best_total.pth")
#model3.optimize_for_inference()

#if torch.cuda.is_available(): model.to('cuda')
if torch.cuda.is_available(): model2.to('cuda')

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

PIXELS_PER_MM = None

class_names = ["Ref Batch"]

def measure_axes_min_area_rect(mask_uint8, PIXELS_PER_MM):
    """
    mask_uint8: (H,W) binary mask with values {0,1} or {0,255}
    Returns: dict with long_axis, short_axis, angle_deg, center, box_points
    """
    # Ensure 0/255 for OpenCV
    if mask_uint8.max() == 1:
        mask = (mask_uint8 * 255).astype(np.uint8)
    else:
        mask = mask_uint8.astype(np.uint8)

    # Get contour (largest if there are multiple)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 5:
        return None

    rect = cv2.minAreaRect(cnt)             # ((cx,cy),(w,h),angle)
    (cx, cy), (w, h), angle = rect
    long_axis = max(w, h)
    short_axis = min(w, h)

    # Convert to physical units if scale known

    if PIXELS_PER_MM:
        long_axis_val = long_axis / PIXELS_PER_MM
        short_axis_val = short_axis / PIXELS_PER_MM
        units = "mm"
    else:
        long_axis_val = long_axis
        short_axis_val = short_axis
        units = "px"

    box = cv2.boxPoints(rect).astype(int)

    return {
        "long_axis": long_axis_val,
        "short_axis": short_axis_val,
        "units": units,
        "angle_deg": angle,        # angle of the rectangle’s width side in OpenCV coords
        "center": (int(cx), int(cy)),
        "box_points": box
    }

def apply_filter(frame):


    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    try:

        result = model2.predict([frame], conf=0.8)
        detections = model3.predict(frame, threshold=0.75)
        
        frame = result[0].plot()

        overlay = frame.copy()

        labels = [
            f"{class_names[class_id]} {confidence:.2f}"
            for class_id, confidence
            in zip(detections.class_id, detections.confidence)
        ]

        bbox_sizes = []
        for xyxy in detections.xyxy:  # assuming detections has .xyxy attribute
            x_min, y_min, x_max, y_max = xyxy
            width = x_max - x_min
            height = y_max - y_min
            bbox_sizes.append((width, height))

        if bbox_sizes:
            PIXELS_PER_MM = (bbox_sizes[0][0]+bbox_sizes[0][1])/20/1.15
        else: 
            PIXELS_PER_MM = None

        if result[0].masks is not None:
            masks = result[0].masks.data.cpu().numpy()  # (N,H,W) in {0,1} floats
            masks = (masks > 0.5).astype(np.uint8)

            for m in masks:
                # draw green translucent mask
                colored_mask = np.zeros_like(frame, dtype=np.uint8)
                colored_mask[m.astype(bool)] = (0, 255, 0)
                overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.4, 0)

                # measure axes

                meas = measure_axes_min_area_rect(m, PIXELS_PER_MM)

                if meas is None:
                    continue

                # draw rotated bounding box (red) and a line along the long axis (cyan)
                cv2.polylines(overlay, [meas["box_points"]], isClosed=True, color=(0, 0, 255), thickness=2)

                # compute and draw the long-axis direction (center -> midpoint of the longer side)
                # find the two long-side midpoints
                box = meas["box_points"]
                # order box points roughly: (tl, tr, br, bl)
                # (cv2.boxPoints returns them in order; we can sort by y then x as a simple heuristic)
                # but for a robust long-axis line, compute side lengths and pick the longer side
                d01 = np.linalg.norm(box[0] - box[1])
                d12 = np.linalg.norm(box[1] - box[2])
                if d01 >= d12:
                    pA, pB = box[0], box[1]
                    pC, pD = box[2], box[3]
                else:
                    pA, pB = box[1], box[2]
                    pC, pD = box[3], box[0]
                mid1 = ((pA[0]+pB[0])//2, (pA[1]+pB[1])//2)
                mid2 = ((pC[0]+pD[0])//2, (pC[1]+pD[1])//2)
                cv2.line(overlay, mid1, mid2, (255, 255, 0), 2)

                # put text near center
                cx, cy = meas["center"]
                txt = f"L: {meas['long_axis']:.1f}{meas['units']}  W: {meas['short_axis']:.1f}{meas['units']}"
                cv2.putText(overlay, txt, (cx+5, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 2)
                cv2.putText(overlay, txt, (cx+5, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    # Draw boxes

            annotated_image = box_annotator.annotate(overlay, detections)
            annotated_image = label_annotator.annotate(annotated_image, detections, labels)

        return overlay
    
    except:
        return frame

