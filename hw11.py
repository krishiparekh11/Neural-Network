"""HW - Chapter 11 “Convolution” 
 Programming Assignment:Implement a convolution algorithm without padding and without striding for original image 6 x 6 and filter 3 x 3. """        
def convolve2d(image, kernel):
    
    n1, n2 = len(image), len(image[0])        # image height, width
    f1, f2 = len(kernel), len(kernel[0])      # kernel height, width

    out_rows = n1 - f1 + 1                    # 6 - 3 + 1 = 4
    out_cols = n2 - f2 + 1                    # 6 - 3 + 1 = 4
    output = [[0] * out_cols for _ in range(out_rows)]

    for i in range(out_rows):                 # slide down, stride 1
        for j in range(out_cols):             # slide right, stride 1
            total = 0
            for u in range(f1):               # element-wise product
                for v in range(f2):           # with the patch under the kernel
                    total += image[i + u][j + v] * kernel[u][v]
            output[i][j] = total              # sum of products
    return output


if __name__ == "__main__":
    image = [
        [3, 0, 1, 2, 7, 4],
        [1, 5, 8, 9, 3, 1],
        [2, 7, 2, 5, 1, 3],
        [0, 1, 3, 1, 7, 8],
        [4, 2, 1, 6, 2, 8],
        [2, 4, 5, 2, 3, 9],
    ]
    kernel = [
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
    ]

    result = convolve2d(image, kernel)
    for row in result:
        print(" ".join(f"{x:4d}" for x in row))