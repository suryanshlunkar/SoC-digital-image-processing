import numpy as np
import cv2 

img = cv2.imread("input_3.jpg")
sigma   = 0.7
t1      = 0.05
t2      = 0.15


# It generates gaussian kernal which is to be convoluted with the grayscale image to reduce noise
# It returns gaussian kernel 
def generate_gaussian_kernel(size, sigma): 
    k = (size - 1) / 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    gaussian_kernel = np.exp(-((x**2 + y**2) / (2*sigma**2)))
    return gaussian_kernel / gaussian_kernel.sum()


def non_maximum_suppression(G, theta):
    # Finding dimensions of the image
    N, M = G.shape
    print(f"Dimensions of the image {N}x{M}")
    # Creating a copy of the gradient magnitude image
    G_suppressed = np.copy(G)

    # Parsing through all pixels
    for i_x in range(M):
        for i_y in range(N):

            grad_ang = theta[i_y, i_x]
            grad_ang = abs(grad_ang - 180) if abs(grad_ang) > 180 else abs(grad_ang)

            # selecting the neighbours of the target pixel
            # according to the gradient direction
            # In the x-axis direction
            if grad_ang <= 22.5 or grad_ang > 157.5:
                neighb_1_x, neighb_1_y = i_x - 1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y

            # top right (diagonal-1) direction
            elif 22.5 < grad_ang <= 67.5:
                neighb_1_x, neighb_1_y = i_x + 1, i_y - 1
                neighb_2_x, neighb_2_y = i_x - 1, i_y + 1

            # In y-axis direction
            elif 67.5 < grad_ang <= 112.5:
                neighb_1_x, neighb_1_y = i_x, i_y - 1
                neighb_2_x, neighb_2_y = i_x, i_y + 1

            # top left (diagonal-2) direction
            elif 112.5 < grad_ang <= 157.5:
                neighb_1_x, neighb_1_y = i_x - 1, i_y - 1
                neighb_2_x, neighb_2_y = i_x + 1, i_y + 1

            # Non-maximum suppression step
            if 0 <= neighb_1_x < M and 0 <= neighb_1_y < N:
                if G[i_y, i_x] < G[neighb_1_y, neighb_1_x]:
                    G_suppressed[i_y, i_x] = 0
                    continue

            if 0 <= neighb_2_x < M and 0 <= neighb_2_y < N:
                if G[i_y, i_x] < G[neighb_2_y, neighb_2_x]:
                    G_suppressed[i_y, i_x] = 0

    return G_suppressed


# Perform hysteresis thresholding to determine strong and weak edges.
def hysteresis_thresholding(img, t1, t2):
    
    weak = np.zeros_like(img)
    strong = np.zeros_like(img)
    strong_threshold = np.max(img) * t2
    weak_threshold = np.max(img) * t1

    strong[img >= strong_threshold] = 255
    weak[(img >= weak_threshold) & (img < strong_threshold)] = 128

    # perform connectivity analysis to determine strong edges
    M, N = img.shape
    edge_map = np.uint8(strong)
    for i in range(1, M-1):
        for j in range(1, N-1):
            if weak[i,j] == 128:
                if (strong[i-1:i+2, j-1:j+2] == 255).any():
                    edge_map[i,j] = 255
                else:
                    edge_map[i,j] = 0

    return edge_map

def apply_convolution(img, kernel):
    M, N = img.shape
    m, n = kernel.shape

    # Calculate the padding size to ensure the output image has the same size as the input image
    pad_x = (m - 1) // 2
    pad_y = (n - 1) // 2

    # Create an empty output image
    output = np.zeros_like(img)

    # Apply convolution to each pixel in the input image
    for i in range(pad_x, M - pad_x):
        for j in range(pad_y, N - pad_y):
            # Extract the region of interest in the input image
            region = img[i - pad_x: i + pad_x + 1, j - pad_y: j + pad_y + 1]

            # Perform element-wise multiplication between the region and the kernel
            conv_result = region * kernel

            # Sum the results to get the final output value for the current pixel
            output[i, j] = conv_result.sum()

    return output

def sobel_op(img):
    dx_kernel = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], dtype=np.float32)

    dy_kernel = np.array([[-1, -2, -1],
                          [0, 0, 0],
                          [1, 2, 1]], dtype=np.float32)

    dx = np.zeros_like(img, dtype=np.float32)
    dy = np.zeros_like(img, dtype=np.float32)

    height, width = img.shape

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Extract the region of interest in the input image
            region = img[i - 1: i + 2, j - 1: j + 2]

            # Perform element-wise multiplication between the region and the x and y Sobel kernels
            dx_result = region * dx_kernel
            dy_result = region * dy_kernel

            # Sum the results to get the final output value for the current pixel
            dx[i, j] = dx_result.sum()
            dy[i, j] = dy_result.sum()

    return dx, dy


# Final canny detector function
def Canny_detector(img, sigma, t1, t2):
    # Step 1: Convert given image to grayscale image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img_gray
    
    # Step 2: Apply Gaussian filter to smooth the image
    kernel_size = int(2 * round(3 * sigma) + 1)
    gaussian_kernel = generate_gaussian_kernel(kernel_size, sigma)   
    img_smooth = apply_convolution(img, gaussian_kernel)
    smooth_img = img_smooth
    
    # Step 3: Compute gradient  Gnitude and direction using Sobel operators
    gx, gy = sobel_op(img)
    G_mag, G_dir = cv2.cartToPolar(gx, gy, angleInDegrees = True)

    # Step 4: Perform non-maximum suppression to thin the edges
    G_suppressed = non_maximum_suppression(G_mag, G_dir)

    # Step 5: Perform hysteresis thresholding to detect strong and weak edges
    edge_image= hysteresis_thresholding(G_suppressed, t1, t2)

	# returns grayscale image , smoothened image, edge image
    return img_gray, smooth_img, edge_image




img_gray, img_smooth, img_op = Canny_detector(img, sigma, t1, t2)
cv2.imwrite(f"img_gray.jpeg", img_gray)
cv2.imwrite(f"img_smooth.jpeg", img_smooth)
cv2.imwrite(f"img_op.jpeg", img_op)