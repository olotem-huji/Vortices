import os, sys, json, glob, base64
import numpy as np
import cv2

# === Change this to your directory ===
labelme_dir = r"C:\Physics\Year 2\Lab\Advanced Lab 2B\Data\mid_600 RPM labeled"

# output folder for masks
mask_dir = os.path.join(labelme_dir, "masks")
os.makedirs(mask_dir, exist_ok=True)

json_files = glob.glob(os.path.join(labelme_dir, "*.json"))

for jf in json_files:
    with open(jf, "r") as f:
        data = json.load(f)

    # try to find the corresponding image
    img = None
    if data.get("imagePath"):
        candidate = os.path.join(labelme_dir, data["imagePath"])
        if os.path.exists(candidate):
            img = cv2.imread(candidate)

    # fallback: decode embedded imageData
    if img is None and data.get("imageData"):
        arr = np.frombuffer(base64.b64decode(data["imageData"]), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        print("⚠️ Could not find image for", jf, "- skipping")
        continue

    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # draw shapes
    for shape in data.get("shapes", []):
        pts = np.array(shape["points"], dtype=np.int32)
        stype = shape.get("shape_type", "polygon")
        if stype in ["polygon", "rectangle"]:
            cv2.fillPoly(mask, [pts], 255)
        elif stype in ["line", "polyline"]:
            cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=2)
            mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)
        else:
            cv2.fillPoly(mask, [pts], 255)

    out_path = os.path.join(mask_dir, os.path.splitext(os.path.basename(jf))[0] + ".png")
    cv2.imwrite(out_path, mask)
    print("✔️ Wrote", out_path)

print("All done! Masks saved in:", mask_dir)
