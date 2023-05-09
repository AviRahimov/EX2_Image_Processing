import math
import numpy as np
import cv2


def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return int(214423147)


def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    """
    The formula for calculating 1D convolution.
    ______________________________________________
    ||##########################################||
    ||#                    _____               #||
    ||# h[j] = (f*g)[j] =  \\ f[j-i]*g[i]      #||
    ||#                    //___               #||
    ||#             -inf <= i <= inf           #||
    ||##########################################||    
    ||__________________________________________||          
    """
    # Reverse the filter h
    kernel_flipped = np.flip(k_size)

    # Initialize the output y with zeros
    res_arr = np.zeros(len(in_signal) + len(k_size) - 1)

    # Perform the convolution
    for i in range(len(res_arr)):
        for j in range(len(k_size)):
            if 0 <= i - j < len(in_signal):
                res_arr[i] += in_signal[i - j] * kernel_flipped[j]

    return res_arr


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """
    # first, we flip the kernel
    kernel_flipped = np.flip(kernel)
    # calculate the number of pixels to add for each axis since we need to perform the padding like the
    # cv2.BORDER_REPLICATE type so I use the same padding method with edge mode so we can save the edges
    # of rhe original image
    num_of_edge_padding = kernel_flipped.shape[0] // 2
    img_mat = np.pad(in_image, pad_width=num_of_edge_padding, mode='edge')
    res_mat = np.zeros(in_image.shape)
    for i in range(res_mat.shape[0]):
        for j in range(res_mat.shape[1]):
            conv_pixel = 0
            for k in range(kernel_flipped.shape[0]):
                for l in range(kernel_flipped.shape[1]):
                    conv_pixel += img_mat[k+i, l+j] * kernel_flipped[k, l]
            res_mat[i, j] = conv_pixel
    return res_mat


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale image
    :return: (directions, magnitude)
    """
    # derivative vectors
    rows_vec = np.array([1, 0, -1])
    column_vec = rows_vec.T
    # x_derivative = conv2D(in_image, rows_vec)
    # y_derivative = conv2D(in_image, column_vec)
    x_derivative = cv2.filter2D(in_image, -1, rows_vec, borderType=cv2.BORDER_REPLICATE)
    y_derivative = cv2.filter2D(in_image, -1, column_vec, borderType=cv2.BORDER_REPLICATE)
    """
    Magnitude = |∇f| =  √((∂f/∂x)² + (∂f/∂y)²)
    Direction = α = arc-tan((∂f/∂y) / (∂f/∂x))
    """
    magnitude = np.sqrt(np.power(x_derivative, 2) + np.power(y_derivative, 2))
    direction = np.arctan2(y_derivative, x_derivative)
    return direction, magnitude


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    # Create an array of indices centered at zero
    indices = np.arange(k_size) - (k_size - 1) / 2
    sigma = (k_size - 1) / 6
    # Calculating the gaussian formula(1D array) provided in the lectures
    gaussian_formula = np.exp(-indices ** 2 / (2 * sigma ** 2))
    # create the second 1D array
    kernel2 = (1 / (sigma * np.sqrt(2 * np.pi))) * gaussian_formula
    # create the kernel by convolve the two 1D arrays
    gaussian_kernel = np.outer(kernel2, gaussian_formula)
    # normalized the kernel
    gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)
    return conv2D(in_image, gaussian_kernel)


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using Opencv2 built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    if k_size % 2 != 0:
        raise "The kernel size should always be an odd number"

    # the length for 99 percentile of gaussian pdf is 6 sigma
    sigma = (k_size - 1) / 6
    kernel = cv2.getGaussianKernel(k_size, sigma)

    blurred_img = cv2.filter2D(in_image, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    return blurred_img


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """
    # derivative the smoothing image with the laplacian kernel both on x and y directions
    deriv_vector = np.array([1, -2, 1])
    sec_deriv_x = conv2D(img, deriv_vector)
    sec_deriv_y = conv2D(img, deriv_vector.T)
    # adding the derivations and convolve it with the smoothed image
    conv_kernel = sec_deriv_x + sec_deriv_y
    laplacian_img = conv2D(img, conv_kernel)
    # create a threshold by taking the maximum intensity in the laplacian image and divide it by 2
    threshold = max(laplacian_img) / 2
    edges_img = np.zeros_like(laplacian_img)
    # apply threshold to get the edges image
    edges_img[laplacian_img > threshold] = 255
    edges_img[laplacian_img <= threshold] = 0
    return edges_img


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """
    # smoothing the image
    smoothed_img = blurImage2(img, 5)
    return edgeDetectionZeroCrossingSimple(smoothed_img)


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use Opencv2 function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
                [(x,y,radius),(x,y,radius),...]
    """
    # I use cv2.Canny to detect edges in the image but first, I blurred the image with Gaussian blurring
    # and set the threshold that seems right to me after some trial and error.
    blur_img = np.uint8(cv2.GaussianBlur(img, (5, 5), 2.5))
    edges = cv2.Canny(blur_img, threshold1=50, threshold2=150)
    # Initialize accumulator array
    accumulator = np.zeros((img.shape[0], img.shape[1], max_radius - min_radius + 1), dtype=np.uint64)

    # Define the radius range to be searched
    radius_range = range(min_radius, max_radius + 1)

    # Loop over all radii and vote for each circle
    for r in radius_range:
        # Create a circle template with the current radius
        circle_template = np.zeros((r * 2 + 1, r * 2 + 1), dtype=np.uint8)
        cv2.circle(circle_template, (r, r), r, 255, 1)

        # Perform the Hough Transform for circles of the current radius
        convolved = cv2.filter2D(edges, -1, circle_template)
        accumulator[:, :, r - min_radius] = convolved

    # Find the maximum values in the accumulator array
    max_vals = np.amax(accumulator, axis=2)

    # Find the coordinates and radius of the circles with maximum votes
    circles = []
    for r in radius_range:
        maxima = np.argwhere(accumulator[:, :, r - min_radius] == max_vals)
        for i in range(len(maxima)):
            circles.append((maxima[i][1], maxima[i][0], r))

    return circles

def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: Opencv2 implementation, my implementation
    """
    cv_func = cv2.bilateralFilter(in_image, k_size, sigma_color, sigma_space)
    num_of_pixels_to_pad = k_size // 2
    image_padded = np.pad(in_image, pad_width=num_of_pixels_to_pad, mode='edge')
    bilateral_img = np.zeros_like(in_image)

    # Create a grid of coordinates
    x, y = np.mgrid[-k_size // 2 + 1:k_size // 2 + 1, -k_size // 2 + 1:k_size // 2 + 1]

    # Calculate the spatial component of the weight matrix
    dist_diff = np.exp(- (x ** 2 + y ** 2) / (2 * sigma_space ** 2))

    for i in range(bilateral_img[0]):
        for j in range(bilateral_img[1]):
            # Get the current pixel value
            center = image_padded[i:i + k_size, j:j + k_size]
            # Calculate the intensity component of the weight matrix
            color_diff = np.exp(- (center - in_image[i, j]) ** 2 / (2 * sigma_color ** 2))
            # Calculate the bilateral filter weight matrix
            weight = color_diff * dist_diff

            # Normalize the weight matrix
            weight = weight / np.sum(weight)

            # Calculate the filtered pixel value
            bilateral_img[i, j] = np.sum(center * weight)

    return cv_func, bilateral_img
