import sys
import cv2
sys.path.append('/home/wsw/workplace/deeplearning/DRWord/algorithm/')

from DRWord_en import predict_en
from DRWord_ch import predict_mtwi
#from DRWord_en.predict_en import predicten
#from DRWord_ch.predict_mtwi import predictmtwi
#im_result, txt_result = predicten('/home/wsw/deeplearning/DRWord/algorithm/test/img_55.jpg')
im_result, txt_result = predict_mtwi.predictmtwi('/home/wsw/workplace/deeplearning/DRWord/algorithm/test/img_55.jpg')
print(txt_result)
print(type(im_result))
cv2.imwrite('/home/wsw/workplace/deeplearning/DRWord/algorithm/tset1.jpg', im_result)
