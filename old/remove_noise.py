import cv2
from enum import Enum


class Algorithm(Enum):
    Canny = "Canny"
    Laplace = "Laplace"
    Scharr = "Scharr"
    Morphological = "Morphological"
def detect_edges(gray, algorithm=Algorithm.Canny):
    if algorithm == Algorithm.Canny:
        edges = cv2.Canny(gray, 100, 200)
    elif algorithm == Algorithm.Laplace:
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        edges = cv2.convertScaleAbs(laplacian)
    elif algorithm == Algorithm.Scharr:
        scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        scharr_magnitude = cv2.magnitude(scharrx, scharry)
        edges = cv2.convertScaleAbs(scharr_magnitude)
    elif algorithm == Algorithm.Morphological:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

    return edges

def remove_noise(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply Gaussian blur to remove noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = detect_edges(gray, Algorithm.Canny)
    _, edges = cv2.threshold(edges, thresh=1, maxval=255, type=cv2.THRESH_BINARY)
    cv2.namedWindow("Edges", cv2.WINDOW_NORMAL)
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_path = r"C:\Physics\Year 2\Lab\Advanced Lab 2B\Vortices\Outputs\PXL_20250521_053352973\Frames\Frame_1.jpg"
remove_noise(image_path)
