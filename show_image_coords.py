import cv2
from matplotlib import pyplot as plt 

def main():
    image = cv2.imread('images/camera1_render.png') # binh tra: image[312, 247]
    plt.imshow(image)
    plt.show()

if __name__ == '__main__':
    main()