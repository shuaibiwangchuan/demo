import paddlehub as hub
import matplotlib.pyplot as plt
import cv2
import sys

image_adress = sys.argv[1]
style_adress = '/home/aistudio/work/' + sys.argv[2] + '.jpg'

stylepro_artistic = hub.Module(name="stylepro_artistic")

results = stylepro_artistic.style_transfer(
    images=[{
        'content': cv2.imread(image_adress),
        'styles': [cv2.imread(style_adress)]
    }],
    alpha = 1.0,
    visualization = True,
)

cv2.imwrite('/home/aistudio/transfer.jpg', results[0]['data'])