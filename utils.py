import cv2
import numpy as np


def average_images(image_list):
    if not image_list:
        raise ValueError("No images provided.")

    # Convert first image to float32
    avg = image_list[0].astype(np.float32)

    # Accumulate the sum
    for img in image_list[1:]:
        avg += img.astype(np.float32)

    # Divide by number of images
    avg /= len(image_list)

    # Convert back to uint8
    return cv2.convertScaleAbs(avg)



def create_diff_image(frame, thresh_value = 5):
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


def create_edges_image(img):
    min_width = 5
    jump = 3
    output = np.zeros_like(img)
    last_left, last_right = 0, img.shape[0] - 1
    # print(img[150][0])
    for row_num in range(img.shape[0]):
        if row_num % 6:
            continue
        # left_range = max(last_left - jump, 0)
        # right_range = max(last_right + jump, left_range + 20)
        # non_zeros = np.nonzero(img[row_num][left_range : right_range])[0]
        non_zeros = np.nonzero(img[row_num])[0]
        # if row_num < 150:
        #     continue
        # print(non_zeros)
        # break
        if not len(non_zeros):
            continue
        left_index = non_zeros[0]
        right_index = non_zeros[-1]
        left_index_found = False
        for j, index in enumerate(non_zeros):
            if j+2 >= len(non_zeros):
                break
            if img[row_num][index + 1] and img[row_num][index + 2]:
                if not left_index_found:
                    left_index = index
                    left_index_found = True
                right_index = index + 2
        if len(non_zeros) == 0:
            continue
        #

        # last_left, last_right = left_index, right_index
        output[row_num][left_index] = 255
        output[row_num][right_index] = 255
    return output


def get_func_from_image(image):
    non_zero_points = cv2.findNonZero(image)
    x = [point[0][0] for point in non_zero_points]
    y = [image.shape[1] - point[0][1] for point in non_zero_points]
    # print(x)
    # print(y)
    return x, y


def black_out_area(image, x_start, x_end, y_start, y_end):
    output = image
    output[y_start: y_end, x_start: x_end] = 0
    return output
