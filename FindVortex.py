import imageio
import matplotlib.pyplot as plt
import cv2
import numpy as np

import utils
import fits

thresh_value = 5

COUNTER = 1

# plt.style.use("physrev.mlpstyle")



def crop_image(img):
    rotated = img
    # rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    h, w = rotated.shape[:2]
    top = int(0.40 * h)
    bottom = int(0.90 * h)  # keep 90% ⇒ remove 10%
    left = int(0.25 * w)
    right = int(0.75 * w)  # keep 50% ⇒ remove 25% each side
    return rotated[top:bottom, left:right]



import cv2
import numpy as np

# def isolate_vortex(gray):
#     h, w = gray.shape
#
#     # Step 1: Ignore top third where noise is dominant
#     roi = gray[h//5:, :]
#
#     # Step 2: Contrast enhancement + edge detection
#     norm = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX)
#     blur = cv2.GaussianBlur(norm, (5, 5), 0)
#     edges = cv2.Canny(blur, 50, 150)
#
#     # Step 3: Find contours
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     best_match = None
#     best_score = -np.inf
#
#     for cnt in contours:
#         x, y, cw, ch = cv2.boundingRect(cnt)
#         if ch < 30 or cw < 10:
#             continue  # too small to be the vortex
#
#         cx = x + cw // 2
#         vertical_profile = np.ptp(cnt[:, 0, 0]) / (np.ptp(cnt[:, 0, 1]) + 1e-5)  # width/height
#
#         # Ideal vortex: tall, thin, centered
#         center_score = 1.0 - abs((cx + x) - w // 2) / (w / 2)
#         aspect_score = 1.0 - vertical_profile  # prefer narrow width-to-height ratio
#         score = ch * aspect_score * center_score  # weighted
#
#         if score > best_score:
#             best_score = score
#             best_match = cnt
#
#     # Step 4: Create mask and return
#     mask = np.zeros_like(gray)
#     if best_match is not None:
#         cv2.drawContours(mask[h//3:], [best_match], -1, 255, thickness=cv2.FILLED)
#
#     result = cv2.bitwise_and(gray, gray, mask=mask)
#     return result


import numpy as np
import cv2
def find_top_fuzzy_paths(binary, top_n=3, x_tolerance=10, min_width=3, min_length=20, overlap=False):
    import copy

    height, width = binary.shape
    runs_by_row = []

    # Step 1: Extract horizontal runs
    for y in range(height):
        row = binary[y]
        runs = []
        x = 0
        while x < width:
            if row[x] > 0:
                start = x
                while x < width and row[x] > 0:
                    x += 1
                end = x - 1
                if end - start + 1 >= min_width:
                    runs.append((start, end))
            else:
                x += 1
        runs_by_row.append(runs)

    all_paths = []

    binary_used = binary.copy()

    for y_start in range(height):
        for run in runs_by_row[y_start]:
            x_center = (run[0] + run[1]) // 2
            path = [(y_start, run)]

            y = y_start + 1
            while y < height:
                candidates = []
                for r in runs_by_row[y]:
                    r_center = (r[0] + r[1]) // 2
                    if abs(r_center - x_center) <= x_tolerance:
                        candidates.append(r)

                if not candidates:
                    break

                next_run = min(candidates, key=lambda r: abs((r[0] + r[1]) // 2 - x_center))
                x_center = (next_run[0] + next_run[1]) // 2
                path.append((y, next_run))
                y += 1

            if len(path) >= min_length:
                all_paths.append(path)

    # Step 2: Sort all found paths by length descending
    all_paths = sorted(all_paths, key=lambda p: len(p), reverse=True)

    final_paths = []
    used_mask = np.zeros_like(binary, dtype=bool)

    for path in all_paths:
        if len(final_paths) >= top_n:
            break

        # Skip path if it overlaps with already used pixels (unless overlap is allowed)
        if not overlap:
            overlap_found = False
            for y, (x0, x1) in path:
                if used_mask[y, x0:x1+1].any():
                    overlap_found = True
                    break
            if overlap_found:
                continue

        # Mark used pixels
        for y, (x0, x1) in path:
            used_mask[y, x0:x1+1] = True

        final_paths.append(path)

    # Step 3: Build masks for each path
    masks = []
    for path in final_paths:
        mask = np.zeros_like(binary)
        for y, (x0, x1) in path:
            mask[y, x0:x1 + 1] = 255
        masks.append(mask)

    return masks, final_paths


def find_fuzzy_vertical_path(
    binary,
    x_tolerance=10,
    min_width=3,
    min_length=20,
    direction='neutral',
    max_hop=10,
    max_back_hop=20,
    hop_cooldown=5,
    max_row_skip=10,
    skip_cooldown=5
):
    height, width = binary.shape
    runs_by_row = []

    # 1. Extract horizontal runs
    for y in range(height):
        row = binary[y]
        runs = []
        x = 0
        while x < width:
            if row[x] > 0:
                start = x
                while x < width and row[x] > 0:
                    x += 1
                end = x - 1
                if end - start + 1 >= min_width:
                    runs.append((start, end))
            else:
                x += 1
        runs_by_row.append(runs)

    max_path = []
    max_len = 0

    for y_start in range(height):
        for run in runs_by_row[y_start]:
            path = [(y_start, run)]
            x_center = (run[0] + run[1]) // 2

            y = y_start + 1
            last_opposite_hop = -hop_cooldown
            last_skip_row = -skip_cooldown

            while y < height:
                best_candidate = None
                best_y = None
                best_dx_abs = None
                best_dx = None

                # Check rows from y to y + max_row_skip (skips allowed)
                for check_y in range(y, min(y + max_row_skip + 1, height)):
                    candidates = []
                    for r in runs_by_row[check_y]:
                        r_center = (r[0] + r[1]) // 2
                        dx = r_center - x_center
                        abs_dx = abs(dx)

                        if direction == 'neutral':
                            if abs_dx <= x_tolerance:
                                candidates.append((r, abs_dx, dx))
                        else:
                            if direction == 'right':
                                forward = dx >= 0
                                valid_hop = abs_dx <= (max_hop if forward else max_back_hop)
                                allow_back = not forward and (check_y - last_opposite_hop >= hop_cooldown)
                            elif direction == 'left':
                                forward = dx <= 0
                                valid_hop = abs_dx <= (max_hop if forward else max_back_hop)
                                allow_back = not forward and (check_y - last_opposite_hop >= hop_cooldown)
                            else:
                                valid_hop = abs_dx <= x_tolerance
                                allow_back = True

                            if valid_hop:
                                if forward or allow_back:
                                    candidates.append((r, abs_dx, dx))

                    if not candidates:
                        continue

                    # Pick best candidate for this row (closest x)
                    candidate = min(candidates, key=lambda item: item[1])
                    r, candidate_abs_dx, candidate_dx = candidate

                    # Check if this candidate is better than current best
                    if best_candidate is None or candidate_abs_dx < best_dx_abs:
                        best_candidate = r
                        best_y = check_y
                        best_dx_abs = candidate_abs_dx
                        best_dx = candidate_dx

                if best_candidate is None:
                    break  # no continuation found

                # If we skipped rows, check cooldown
                row_skip = best_y - y
                if row_skip > 1:
                    if (best_y - last_skip_row) < skip_cooldown:
                        break  # can't skip yet, end path
                    last_skip_row = best_y

                # Update opposite hop tracking
                if direction != 'neutral':
                    is_opposite = (direction == 'right' and best_dx < 0) or (direction == 'left' and best_dx > 0)
                    if is_opposite:
                        last_opposite_hop = best_y

                x_center = (best_candidate[0] + best_candidate[1]) // 2
                path.append((best_y, best_candidate))
                y = best_y + 1

            if len(path) > max_len and len(path) >= min_length:
                max_path = path
                max_len = len(path)

    # Build final mask
    mask = np.zeros_like(binary)
    for y, (x0, x1) in max_path:
        mask[y, x0:x1 + 1] = 255

    return mask, max_path





import cv2
import numpy as np

def filter_lines_by_width(binary_img, min_width=3, max_width=50):
    # Ensure binary image
    binary = binary_img.copy()
    if len(binary.shape) > 2:
        binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (min_width <= w <= max_width) or (min_width <= h <= max_width):
            cv2.drawContours(mask, [cnt], -1, 255, cv2.FILLED)

    return mask


def create_diff_image(frame):
    # corrected = crop_image(frame)
    corrected = frame
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray[:, 1:], gray[:, :-1])
    _, binary = cv2.threshold(diff, thresh_value, 200, cv2.THRESH_BINARY)
    denoised = binary
    # denoised = cv2.medianBlur(binary, 5)
    # denoised = cv2.GaussianBlur(denoised, (5, 5), 0)
    # denoised = cv2.bilateralFilter(denoised, 9, 75, 75)

    global COUNTER
    # if COUNTER % 50 == 0:
    cv2.imwrite(fr'./Outputs/Frame_{COUNTER}.jpg', denoised)
    vortex = denoised
    # vortex, _ = find_fuzzy_vertical_path(denoised, direction="right")
    # vortex = filter_lines_by_width(denoised)
    cv2.imwrite("./Vortex.jpg", vortex)
    print(COUNTER)
    COUNTER += 1


# im = cv2.imread(r"C:\Physics\Year 2\Lab\Advanced Lab 2B\Image tests\Test 4.jpg")
# create_diff_image(im)
# cap = cv2.VideoCapture(r"C:\Physics\Year 2\Lab\Advanced Lab 2B\21.05\PXL_20250521_055911851.mp4")
# i = 0
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#     i += 1
#     # if i != 800:
#     #     continue
#     cv2.imwrite(fr'./Outputs/Frame_{i}.jpg', frame)
#     create_diff_image(frame)
# cap.release()

for i in range(1, 2601):
    if i % 15:
        continue
    image_list = [cv2.imread(fr'./Outputs/Frame_{i+j}.jpg', cv2.IMREAD_GRAYSCALE) for j in range(0,15)]
    avg_img = utils.average_images(image_list)
    avg_img[avg_img > 150] = 0
    cv2.imwrite("./avg.jpg", avg_img)
    # avg_img = cv2.imread("./only_edges_manual.jpg", cv2.IMREAD_GRAYSCALE)
    # cropped = avg_img
    rotated = cv2.rotate(avg_img, cv2.ROTATE_90_CLOCKWISE)
    [height, width] = rotated.shape
    cropped = rotated[int(height*0.37):int(height*0.7), int(width*0.12):int(width*0.88)]
    # cropped = rotated[int(height*0.7): int(height*1.4), int(width * 0.06):int(width * 0.45)]
    cv2.imwrite("./cropped.jpg", cropped)
    [height, width] = cropped.shape
    # blacked_out = cropped
    mask = cv2.imread("mask.jpg", cv2.IMREAD_GRAYSCALE)
    blacked_out = cv2.bitwise_and(cropped, cropped, mask=mask)
    # blacked_out = utils.black_out_area(blacked_out, 280, width, height-200, height)
    # blacked_out = utils.black_out_area(blacked_out, 300, width, height - 260, height)
    only_edges = utils.create_edges_image(blacked_out)
    cv2.imwrite("./only_edges.jpg", only_edges)
    # plt.imshow(cropped, cmap="gray")
    # plt.show()
    x, y = utils.get_func_from_image(only_edges)
    left_x = []
    left_y = []
    right_x = []
    right_y = []

    for j in range(len(x)):
        if x[j] <= 40:
            continue
        if x[j] >= (max(x) + min(x))/2:
            right_x.append(x[j])
            right_y.append(y[j])
        else:
            left_x.append(x[j])
            left_y.append(y[j])


    # plt.scatter(x, y)
    # plt.title("Vortex")
    # plt.show()
    # fits.get_fit(new_x, new_y)
    fits.get_fit(right_x, right_y, side="right")
    fits.get_fit(left_x, left_y, side="left")
    plt.scatter(x, y)
    plt.savefig(fr"./Outputs/Figures/{i} to {i+14}")
    # plt.show()
    plt.clf()
    # manual = cv2.imread("./only_edges_manual.jpg", cv2.IMREAD_GRAYSCALE)
    # x, y = utils.get_func_from_image(manual)
    # fits.get_fit(x, y)
