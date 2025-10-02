import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import fits

scale_mm_per_pixel = 180 / 1080 * 6

heights = {}
all_heights = {}

r_0_short = 0.02
r_0_med = 0.031
r_0_long = 0.0375
r_0 = {"short": r_0_short, "medium": r_0_med, "long": r_0_long}


def sort_for_plot(ys, xs):
    top_points = {}
    for x, y in zip(xs, ys):
        if x not in top_points or y < top_points[x]:
            top_points[x] = y  # keep only smallest y for this x

    # Sort by x
    sorted_xs = sorted(top_points)
    sorted_ys = [top_points[x] for x in sorted_xs]

    return np.array(sorted_ys), np.array(sorted_xs)

dir_path = r"C:\Physics\Year 2\Lab\Advanced Lab 2B\21.05\Traced Vortices\\"
filenames = os.listdir(dir_path)

short = [dir_path + fn for fn in filenames if "short" in fn]
medium = [dir_path + fn for fn in filenames if "mid" in fn]
long = [dir_path + fn for fn in filenames if "long" in fn]

for s in long:
    rpm = [float(re.search(r'\d+\.?\d*', _).group()) for _ in s.split("_") if "RPM" in _][0]
    if rpm != 337:
        continue
    img = cv2.imread(s)
    binary = np.where((img[..., 0] == img[..., 1]) & (img[..., 1] == img[..., 2]), 0, 1)
    ys, xs = np.nonzero(binary)
    ys, xs = sort_for_plot(ys, xs)
    xs = xs * scale_mm_per_pixel
    xs = xs - np.average(xs)
    ys = ys * scale_mm_per_pixel

    # Plot as scatter
    # plt.figure(figsize=(4, 4))

    heights[rpm] = max(ys) - min(ys)
    paddle = "short" if s in short else "medium" if s in medium else "long"
    all_heights.setdefault(paddle, {})[rpm] = max(ys) - min(ys)
    plt.plot(xs, ys-400, linewidth=3, label=f"{rpm} RPM")
    w = (rpm / 60) * (2 * np.pi)
    rankine_ys = fits.rankine_in_cylinder(xs * 1e-3, w, r_max=0.09, r_p=r_0[paddle]) * 1e3
    rankine_ys = rankine_ys - min(rankine_ys)
    rankine_ys += 400
    plt.plot(xs, rankine_ys-400)
plt.gca().invert_yaxis()  # Optional: to match image orientation
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.legend()
plt.title(f"Vortices - {paddle.capitalize()} Paddle")
plt.show()


for k, v in all_heights.items():
    plt.scatter(v.keys(), v.values(), label=f"{k} paddle")
plt.scatter(heights.keys(), heights.values())
plt.xlabel("RPM")
plt.ylabel("Height [mm]")
plt.legend()
plt.title("Vortex Heights - All")
plt.show()
# plt.show()
