import multipriority
import queue
import threading
import numpy as np
from multipriority.feedback import Feedback

import rospy
from multipriority_ros.srv import FeedbackService, FeedbackServiceResponse

if __name__ == "__main__":
    rospy.init_node("feedback_client")
    filepath = "/".join(multipriority.__path__[0].split('/')[:-1])
    body_discretization = {"head": 0, "left arm": 1, "right arm":2, "torso":3, "left leg":4, "right leg":5 }
    input_queue = queue.Queue()
    fb = Feedback(filepath, body_discretization, input_queue)

    while True:
        try:
            try:
                out = fb.input_queue.get(timeout=3)
                res = fb.response(out)
                feedback = fb.process_input(res)
                print("feedback in try is: ", feedback)

                if isinstance(feedback, list):
                    fb.feedback = np.full(len(fb.body_discretization), 3)
                    for pr in feedback:
                        idx = fb.body_discretization[pr[0]]
                        fb.feedback[idx] = pr[1]

                    print(fb.feedback)
                    rospy.wait_for_service('feedback_watcher')
                    try:
                        send_feedback = rospy.ServiceProxy('feedback_watcher', FeedbackService)
                        resp1 = send_feedback(fb.feedback)
                    except rospy.ServiceException as e:
                        print("Service call failed: %s"%e)

                else:
                    feedback, part = feedback
                    if feedback == "specify-left-right":
                        fb.speak(f"Please specify 'left' or 'right' for the body part: {part}.")
                        continue
                    elif feedback == "unrecognized":
                        fb.speak(f"Unrecognized body part: {part}. Please use different terms.")
                        continue

            except queue.Empty:
                # Set feedback to default value if no input is received
                # self.feedback = np.full(len(self.body_discretization), 3)
                # self.publish()
                print("No new input. Nothing is sent.")

        except KeyboardInterrupt:
            print("Interrupted by user, exiting.")
            break
    
    rospy.spin()
