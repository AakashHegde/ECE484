from ultralytics import YOLO
from sensor_msgs.msg import Image
import rospy
from std_msgs.msg import Float32MultiArray, Bool
import cv2
from cv_bridge import CvBridge, CvBridgeError


class ObjectDetection():
    
    def __init__(self):
        self.bridge = CvBridge()
        self.model = YOLO("yolov8n.pt")
        self.sub_image = rospy.Subscriber('/zed2/zed_node/rgb_raw/image_raw_color', Image, self.img_callback, queue_size=1)
        self.pub_obj_det_status = rospy.Publisher("object_detection/detection_status", Bool, queue_size=1)
        self.pub_obj_img = rospy.Publisher("object_detection/object_img", Image, queue_size=1)

        # Object detected : area of bounding box in pixel square units
        # This will determine how close we get to the object before taking action
        self.objects_to_area = {'stop sign': 3000, 'person': 4000, 'car': 7000}
    
    def img_callback(self, data):
        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        raw_img = cv_image.copy()

        results = self.model(raw_img, conf=0.6, verbose=False)

        obj_det_status = Bool(False)
        for result in results:
            for box in result.boxes:
                object_id = self.model.names[box.cls.tolist()[0]]
                if(object_id in list(self.objects_to_area.keys())):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    bbox_area = (x2 - x1) * (y2 - y1)
                    # print(object_id, bbox_area)
                    if(bbox_area >= self.objects_to_area[object_id]):
                        # print(object_id, bbox_area)
                        obj_det_status = Bool(True)
                        
            img = self.bridge.cv2_to_imgmsg(result.plot(), 'bgr8')
            self.pub_obj_img.publish(img)
        self.pub_obj_det_status.publish(obj_det_status)

if __name__ == '__main__':
    # init args
    rospy.init_node('yolo_node', anonymous=True)
    ObjectDetection()
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(1)