import queue
import threading
import numpy as np

import rospy
from multipriority_ros.srv import FeedbackService, FeedbackServiceResponse


def dummy_feedback_server(req):
    # Should be implemented in a similar way for the environment class
    print ("Feedback received: ", req.input_data)
    return FeedbackServiceResponse("success")

if __name__ == "__main__":
    rospy.init_node('feedback_server')
    s = rospy.Service('feedback_watcher', FeedbackService, dummy_feedback_server)
    print("Ready to receive feedback.")
    rospy.spin()