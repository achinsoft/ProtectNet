
##################### ProtectNet Solution ############################
############ by Shaikh Wasim Raja for IBM Code for Call ##############
############ E-mail : shaikraj@in.ibm.com               ##############
#                                                                    #
#################### Dependacy Libraries #############################
############  python -m pip install tensorflow==1.14      ############
############  python -m pip install opencv-python         ############
############  python -m pip install numpy                 ############
######################################################################

import os
import numpy as np
import tensorflow as tf
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class DetectorAPI:
    """
    DetectorAPI Class
    """
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.compat.v1.Session(graph=self.detection_graph)

        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def pr_fr(self, image):
        """
        pr_fr function.
        Input self, image
        """
       
        image_np_expanded = np.expand_dims(image, axis=0)
     
        (boxes_inner, scores_inner, classes_inner, num_inner) = self.sess.run(
            [self.detection_boxes, self.detection_scores, \
                self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
    

        im_height, im_width, _ = image.shape
        boxes_list = [None for j in range(boxes_inner.shape[1])]
        for j in range(boxes_inner.shape[1]):
            boxes_list[j] = (int(boxes_inner[0, j, 0] * im_height),
                             int(boxes_inner[0, j, 1] * im_width),
                             int(boxes_inner[0, j, 2] * im_height),
                             int(boxes_inner[0, j, 3] * im_width))
                     

        return boxes_list, scores_inner[0].tolist(), \
            [int(x) for x in classes_inner[0].tolist()], int(num_inner[0])

    def close(self):
        """
        close method
        """
        self.sess.close()
        self.default_graph.close()


if __name__ == "__main__":
    MODEL_PATH = 'rcnn\\frozen_inference_graph.pb'
    ODAPI = DetectorAPI(path_to_ckpt=MODEL_PATH)
    THRESHOLD = 0.7
    CAP = cv2.VideoCapture(0)

    while True:
        R, IMG = CAP.read()
        IMG = cv2.resize(IMG, (640, 480))
        BOXES, SCORES, CLASSES, NUM = ODAPI.pr_fr(IMG)
        centers = [] 
        for i, _ in enumerate(BOXES):
            if CLASSES[i] == 1 and SCORES[i] > THRESHOLD:
                box = BOXES[i]
                x, y, w, h = box
                img = cv2.rectangle(IMG, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)
                cx1 = x
                cx2 = x+w
                cy1 = y
                cy2 = y + h
                cx = x + (cx2/2)
                cy = y + (cy2/2)
                centers.append([cx, cy])
       
        if len(centers) >= 2:
            D = centers[0][0] - centers[1][0]
            if D < 0:
                D = D*(-1)

            if D < 20:
                print(f"Social distance is broken")

            else:
                print(f"Social distance is OK")    

        cv2.imshow("preview", IMG)
        KEY = cv2.waitKey(1)
        if KEY & 0xFF == ord('q'):
            break
