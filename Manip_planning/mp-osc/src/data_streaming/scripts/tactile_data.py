#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32MultiArray
import threading

class Data:
    def __init__(self):
        rospy.init_node('calibration_subscriber', anonymous=True)
        self.data = None
        self.lock = threading.Lock()
        self.sub = rospy.Subscriber("/calibration", Float32MultiArray, self.callback)

    def callback(self, data):
        with self.lock:
            self.data = data.data
            rospy.loginfo("Received calibration data: %s", self.data)

    def get_latest_data(self):
        with self.lock:
            return self.data

if __name__ == "__main__":
    cal_subscriber = Data()
    rospy.spin()
