import cv2

image = cv2.imread('/home/lkk/code/my_faster/model/image1.jpg')
image = cv2.rectangle(
                image, (50,50), (200,200), (0,255,0), 2
            )
cv2.imwrite('c.jpg',image)