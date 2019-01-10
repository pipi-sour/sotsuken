import cv2
import numpy as np

if __name__ == '__main__':
  img_src = cv2.imread("test.png", 1)
  img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

  thresh = 100
  max_pix = 255
  ret, img_dst = cv2.threshold(img_gray, thresh, max_pix, cv2.THRESH_BINARY)
  cv2.imshow("Show the binarization image", img_dst)
  cv2.imwrite('test_gray.png', img_dst)
  cv2.waitKey(0)
  cv2.destroyAllWindows()