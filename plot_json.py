import json
import matplotlib.pyplot as plt

# Path to your labelme JSON file
json_path = r"C:\Physics\Year 2\Lab\Advanced Lab 2B\Frames\long_337\jsons/Frame_1.json"

# Number of last points to ignore (set to whatever makes sense for your data)
IGNORE_LAST = 9


with open(json_path, "r") as f:
    data = json.load(f)

plt.figure(figsize=(6, 6))

for shape in data["shapes"]:
    points = shape["points"]
    label = shape.get("label", "shape")

    # Ignore the last N points
    if IGNORE_LAST > 0:
        points = points[:-IGNORE_LAST]

    # Extract coordinates
    center_r = (points[0][0] + points[-1][0]) / 2
    r = [p[0] - center_r for p in points]
    z = [p[1] for p in points]

    # Plot as open polyline (no closing)
    plt.plot(r, z, label=label)

plt.gca().invert_yaxis()  # Match LabelMe coordinate system
plt.xlabel("r")
plt.ylabel("z")
plt.title("Open Polygon from LabelMe")
plt.legend()
plt.axis("equal")
plt.show()