from multipriority.utils import *
import numpy as np
import yaml
import argparse
from pprint import pformat


parser = argparse.ArgumentParser()
parser.add_argument('--num_users', type=int, default=5)

args = parser.parse_args()

num_users = args.num_users

# Generate user profiles
users = {f"user_{i}": generate_user_with_unique_priority() for i in range(num_users)}

# Save user profiles
save_yaml(f'test_user_profiles_{num_users}_users.yaml', users)

# Test user profiles
loaded_users = load_yaml(f'user_profiles_{num_users}_users.yaml')

user = loaded_users['user_0']
print ("User: \n", pformat(user))
# Generatea a random sorted active dict
body_parts = user.keys()
sorted_active_dict = {}
for body_part in body_parts:
    sorted_active_dict[body_part] = np.array([1020, np.random.uniform(0, 7)])

print ("\nSorted active dict: \n", pformat(sorted_active_dict))
# Simulate user feedback
user_feedback = simulate_feedback(sorted_active_dict, user)

print ("\nOutput feedback: \n", pformat(user_feedback))