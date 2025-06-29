#!/usr/bin/env python3

import os
import queue
import re
import subprocess
import threading
from pathlib import Path

import numpy as np
import yaml
from .detect_wakeword import DetectWord
from openai import OpenAI
from .speech_to_text import Speech

import multipriority


class Feedback:
    """
    A class to handle feedback processing for adjusting force applied to various body parts.
    
    Attributes:
        client (OpenAI): An instance of the OpenAI client.
        body_discretization (dict): A mapping of body part names to indices.
        feedback (numpy.ndarray)rcare_py/pyrcareworld/Examples/feedback_audio: An array representing feedback values for each body part.
        body_parts (dict): A dictionary mapping body part names to their corresponding group names.
        valid_matches (list): A list to store valid body part and scale pairs from user input.
    
    Methods:
        get_input_audio(): Retrieves audio input from the user and processes it to text.
        load_body_parts(file_path): Loads body part data from a YAML file and creates an inverted index.
        response(input): Sends a user's input to OpenAI's GPT model and retrieves the response.
        parse_response_to_pairs(response_object): Parses the GPT model's response into body part and scale pairs.
        process_input(response_object): Processes the GPT model's response and updates valid_matches with valid pairs.
        main(): The main loop for obtaining user input, processing it, and updating feedback values.
    """
    def __init__(self, filepath, body_discretization, input_queue):
        """
        Initializes the Feedback object with necessary attributes.

        Parameters:
            filepath (str): The file path to the directory containing relevant files.
            body_discretization (dict): A mapping of body part names to indices.
        """
        # Initialize Feedback
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
        self.body_discretization = body_discretization
        self.feedback = np.full(len(self.body_discretization), 3)
        self.body_parts = self.load_body_parts(filepath + "/configs/body_parts.yaml")
        self.valid_matches = []
        self.input_queue = input_queue

        # Initialize DetectWord
        audio_path = "/tmp/feedback_audio"
        if not os.path.exists(audio_path):
            os.makedirs(audio_path)
        keywords = ["jarvis", "thank you"]
        access_key = os.environ.get("PORCUPINE_API_KEY")
        self.detect_word_instance = DetectWord(
            access_key=access_key,
            keywords=keywords,
            audio_directory=audio_path,
            update_event=threading.Event(),
            filepath=filepath
        )

        # Initialize Speech
        model_name = "base"
        self.speech_instance = Speech(model_name, audio_path, self.detect_word_instance.update_event, input_queue)

        # Start DetectWord and Speech in separate threads
        self.detect_word_thread = threading.Thread(target=self.detect_word_instance.main)
        self.detect_word_thread.start()

        self.speech_thread = threading.Thread(target=self.speech_instance.main, args=(self.detect_word_instance,))
        self.speech_thread.start()


    def load_body_parts(self, file_path):
        """
        Loads body part data from a YAML file and creates an inverted index dictionary for efficient lookups.

        Parameters:
            file_path (str): The file path to the YAML file containing body part data.

        Returns:
            dict: An inverted index dictionary mapping each body part to its corresponding group.
        """
        with open(file_path, 'r') as file:
            body_parts_data = yaml.safe_load(file)
        inverted_index = {}
        for group, parts in body_parts_data.items():
            for part in parts:
                inverted_index[part.lower()] = group.lower()

        return inverted_index
    
    def speak(self, text):
        """
        Uses espeak to convert text to speech and output it through the speakers.

        Parameters:
            text (str): The text to be spoken.
        """
        try:
            subprocess.run(['espeak', text], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error in text-to-speech conversion: {e}")

    def response(self, input):
        """
        Sends the user input to the OpenAI GPT model and retrieves the model's response.

        Parameters:
            input (str): The user input to be sent to the model.

        Returns:
            openai.Completion: The response object from the GPT model.
        """
        response = self.client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a caregiver assistant programmed to understand specific sentiments about force levels and identify body parts. Your task is to interpret phrases indicating the need for adjustment in the force applied to different body parts. Here is the scale you must use: \
            - 'a lot more', 'much more', 'significantly increase' are mapped to 5, indicating a high increase in force.\
            - 'increase', 'a bit more', 'slightly more' are mapped to 4, indicating a moderate increase in force. \
            - 'neutral', 'no change', 'stay the same' are mapped to 3, indicating no change in force.\
            - 'less', 'a bit less', 'slightly decrease' are mapped to 2, indicating a moderate decrease in force.\
            - 'much less', 'a lot less', 'significantly decrease' are mapped to 1, indicating a high decrease in force.\
            - Generally, 3 is never used\
            \
            Your responses should be in the format of (body part, scale number). For example:\
            \
            - If a patient says, 'There is too much force on my right arm', you should interpret this as ('right arm', 1), because 'too much force' indicates a need for a high decrease in force. \
            - If a patient says, 'Decrease the force on my left leg a bit', interpret this as ('left leg', 2), as 'decrease a bit' suggests a moderate decrease in force. \
            - If a patient says, 'Can you add a bit more force to my midsection?', this should be interpreted as ('midsection', 4), as 'a bit more force' suggests a moderate increase in force. \
            - If a patient says, 'Can you increase the force on my left arm?', this should be interpreted as ('left arm', 4), as 'increase' itself without adverbs suggests a moderate increase in force.\
            - If a patient says, 'There should be much more force on my left arm', interpret this as ('left arm', 5), as 'much more force' indicates a high increase in force. \
                \
            Please analyze the following statement and provide an interpretation based on these guidelines:"},
            {"role": "user", "content": "There is too much force on my right arm"},
            {"role": "assistant", "content": "(\"right arm\", 1)"},
            {"role": "user", "content": input}
        ]
        )
        return response

    def parse_response_to_pairs(self, response_object):
        """
        Parses the GPT model's response into body part and scale pairs.

        Parameters:
            response_object (openai.Completion): The response object from the GPT model.

        Returns:
            list: A list of valid body part and scale pairs.
        """
        response_content = response_object.choices[0].message.content

        pattern = r'\("(.*?)", (\d)\)'
        matches = re.findall(pattern, response_content)
        print(matches)
        valid_parts = []
        for part, scale in matches:
            part_lower = part.lower()
            if part_lower in self.body_parts:
                valid_parts.append((part, int(scale)))
            elif any(bp.startswith(part_lower) for bp in self.body_parts if 'left' in bp or 'right' in bp):
                print(f"Please specify 'left' or 'right' for the body part: {part}.")
                return []
            else:
                print(f"Unrecognized body part: {part}. Please use different terms.")
                return []
            
        return valid_parts

    def process_input(self, response_object):
        """
        Processes the GPT model's response, updates valid_matches with valid pairs, and handles unrecognized inputs.

        Parameters:
            response_object (openai.Completion): The response object from the GPT model.

        Returns:
            list or tuple: A list of valid matches or a tuple indicating an issue and the problematic part.
        """
        response_content = response_object.choices[0].message.content

        pattern = r'\("(.*?)", (\d)\)'
        matches = re.findall(pattern, response_content)
        print(matches)

        for part, scale in matches:
            part_lower = part.lower()

            if part_lower in self.body_parts:
                self.valid_matches.append((self.body_parts[part_lower], scale))
            elif any(part_lower in p for p in self.body_parts if 'left' in p or 'right' in p):
                return "specify-left-right", part
            else:
                return "unrecognized", part

        return self.valid_matches
    
    # def publish(self):
    #     self.feedback = np.array(self.feedback, dtype=float)
    #     feedback_msg = Float32MultiArray(data=self.feedback)
    #     self.feedback_pub.publish(feedback_msg)
    #     print("Published updated feedback to ROS topic")
    
    def main(self):
        """
        The main loop for obtaining user input, processing it with the OpenAI GPT model, and updating feedback values.

        Returns:
            numpy.ndarray or None: An array representing feedback values for each body part, or None if an error occurs.
        """
        while True:
            try:
                try:
                    out = self.input_queue.get(timeout=3)
                    res = self.response(out)
                    feedback = self.process_input(res)
                    print("feedback in try is: ", feedback)

                    if isinstance(feedback, list):
                        self.feedback = np.full(len(self.body_discretization), 3)
                        for pr in feedback:
                            idx = self.body_discretization[pr[0]]
                            self.feedback[idx] = pr[1]

                        print(self.feedback)

                    else:
                        feedback, part = feedback
                        if feedback == "specify-left-right":
                            self.speak(f"Please specify 'left' or 'right' for the body part: {part}.")
                            continue
                        elif feedback == "unrecognized":
                            self.speak(f"Unrecognized body part: {part}. Please use different terms.")
                            continue

                except queue.Empty:
                    # Set feedback to default value if no input is received
                    # self.feedback = np.full(len(self.body_discretization), 3)
                    # self.publish()
                    print("No new input. Nothing is sent.")

            except KeyboardInterrupt:
                print("Interrupted by user, exiting.")
                break
    
if __name__ == "__main__":
    filepath = "/".join(multipriority.__path__[0].split('/')[:-1])
    body_discretization = {"head": 0, "left arm": 1, "right arm":2, "torso":3, "left leg":4, "right leg":5 }
    input_queue = queue.Queue()
    fb = Feedback(filepath, body_discretization, input_queue)
    feedback_thread = threading.Thread(target=fb.main)
    feedback_thread.start()
