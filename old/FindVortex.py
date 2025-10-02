import os

import matplotlib.pyplot as plt

import utils
import fits

thresh_value = 5

COUNTER = 1

# video_name = "PXL_20250521_055911851"

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
    cv2.imwrite("Vortex.jpg", vortex)
    print(COUNTER)
    COUNTER += 1


# im = cv2.imread(r"C:\Physics\Year 2\Lab\Advanced Lab 2B\Image tests\Test 4.jpg")
# create_diff_image(im)


def mask_directory(dir_path, mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    out_dir = fr"{dir_path}\mask"
    os.makedirs(out_dir, exist_ok=True)
    dir_files = os.listdir(dir_path)
    frame_paths = [fr"{dir_path}\{df}" for df in dir_files if ".jpg" in df]
    for i, fp in enumerate(frame_paths):
        frame = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        masked = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imwrite(fr"{out_dir}\{dir_files[i]}", masked)



def average_directory(dir_path, num_per_average, threshold=0 ,rotate=False):
    dir_files = os.listdir(dir_path)
    frame_paths = [fr"{dir_path}\{df}" for df in dir_files if ".jpg" in df]

    out_dir = fr"{dir_path}\averages"
    os.makedirs(out_dir, exist_ok=True)
    images_to_average = []
    for i, frame in enumerate(frame_paths, 1):
        images_to_average.append(cv2.imread(frame, cv2.IMREAD_GRAYSCALE))
        if i % num_per_average == 0:
            avg_img = utils.average_images(images_to_average)
            avg_img[avg_img > 150] = 0
            if rotate:
                avg_img = cv2.rotate(avg_img, cv2.ROTATE_90_CLOCKWISE)
            if threshold:
                avg_img[avg_img < threshold] = 0
                # _, avg_img = cv2.threshold(avg_img, threshold, 255, cv2.THRE)
            cv2.imwrite(fr"{out_dir}\{i-num_per_average}-{i}.jpg", avg_img)
            images_to_average = []




def fit_directory(dir_path):
    image_list = [cv2.imread(fr'{dir_path}/Frame_{100 + j}.jpg', cv2.IMREAD_GRAYSCALE) for j in range(0, 15)]
    avg_img = utils.average_images(image_list)
    avg_img[avg_img > 150] = 0
    avg_img = cv2.rotate(avg_img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite("avg.jpg", avg_img)

    num_of_frames = len([f for f in os.listdir(f"./Outputs/{video_name}/") if "Frame" in f])
    for i in range(1, num_of_frames-14):
        avg_num = 5
        if i % 5:
            continue
        image_list = [cv2.imread(fr'{dir_path}/Frame_{i+j}.jpg', cv2.IMREAD_GRAYSCALE) for j in range(0, avg_num)]
        avg_img = utils.average_images(image_list)
        avg_img[avg_img > 150] = 0
        cv2.imwrite("avg.jpg", avg_img)
        # avg_img = cv2.imread("./only_edges_manual.jpg", cv2.IMREAD_GRAYSCALE)
        # cropped = avg_img
        rotated = cv2.rotate(avg_img, cv2.ROTATE_90_CLOCKWISE)
        # rotated = avg_img
        [height, width] = rotated.shape
        cropped = rotated
        cropped = rotated[int(height*0.37):int(height*0.7), int(width*0.12):int(width*0.88)]
        # cropped = rotated[int(height*0.7): int(height*1.4), int(width * 0.06):int(width * 0.45)]
        cv2.imwrite("cropped.jpg", cropped)
        [height, width] = cropped.shape
        # blacked_out = cropped
        mask = cv2.imread(f"mask {video_name}.jpg", cv2.IMREAD_GRAYSCALE)
        blacked_out = cv2.bitwise_and(cropped, cropped, mask=mask)
        cv2.imwrite(f"{dir_path}/Blacked Out/{i} to {i+avg_num}.jpg", blacked_out)
        # blacked_out = utils.black_out_area(blacked_out, 280, width, height-200, height)
        # blacked_out = utils.black_out_area(blacked_out, 300, width, height - 260, height)
        only_edges = utils.create_edges_image(blacked_out)
        cv2.imwrite("only_edges.jpg", only_edges)
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
        plt.savefig(fr"./Outputs/{video_name}/Figures/{i} to {i+avg_num}.jpg")
        # plt.show()
        plt.clf()
        # manual = cv2.imread("./only_edges_manual.jpg", cv2.IMREAD_GRAYSCALE)
        # x, y = utils.get_func_from_image(manual)
        # fits.get_fit(x, y)


dir_path = r"C:\Physics\Year 2\Lab\Advanced Lab 2B\Vortices\Outputs\PXL_20250521_060205085"
# average_directory(dir_path, 5, threshold=100, rotate=True)
mask_path = r"C:\Physics\Year 2\Lab\Advanced Lab 2B\Vortices\Outputs\PXL_20250521_060205085\averages\mask\mask.jpg"
mask_directory(f"{dir_path}/averages", mask_path)

# video_path = r"C:\Physics\Year 2\Lab\Advanced Lab 2B\21.05\PXL_20250521_053352973.mp4"
# fit_directory(dir_path)
# split_video_to_frames(video_path)


# video_dir = r"C:\Physics\Year 2\Lab\Advanced Lab 2B\21.05"
# vid_names = os.listdir(video_dir)
# vid_paths = [fr"{video_dir}\{vn}" for vn in vid_names]
# for vp in vid_paths:
#     split_video_to_frames(vp)
