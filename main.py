import argparse
import json
from train import train

parser = argparse.ArgumentParser()
parser.add_argument('my_dict', type=str)
args = parser.parse_args()

with open(args.my_dict, 'r') as f:
    my_dictionary = json.load(f)

train(my_dictionary)