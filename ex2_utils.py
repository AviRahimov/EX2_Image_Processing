import math
import numpy as np
import cv2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 214423147


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


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale image
    :return: (directions, magnitude)
    """

    """
    Magnitude = |∇f| =  √((∂f/∂x)² + (∂f/∂y)²)
    Direction = α = arc-tan((∂f/∂y) / (∂f/∂x))
    """


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using Opencv2 built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use Opencv2 function: cv22.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
                [(x,y,radius),(x,y,radius),...]
    """


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: Opencv2 implementation, my implementation
    """
