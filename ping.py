import os
import json
import torch
import random
import requests

PROJ_PATH = 'project/serving/torchServe/test'
API_URL = "http://0.0.0.0:8080/predictions/test-model"

with open(os.path.join(PROJ_PATH, 'user2idx.json'), 'r') as json_file:
    user2idx = json.load(json_file)

with open(os.path.join(PROJ_PATH, 'question2idx.json'), 'r') as json_file:
    question2idx = json.load(json_file)


def gen_samples(user2idx, question2idx, k):
    temp = {
        'userId': random.sample(list(user2idx.keys()), k=k),
        'questionId': random.sample(list(question2idx.keys()), k=k)
    }
    return temp


def query(payload):
    response = requests.post(API_URL, json=payload)
    return response


test_sample = gen_samples(user2idx, question2idx, k=10)
output = query(test_sample)

print(output.json())
