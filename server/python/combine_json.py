#!/usr/bin/python3
# combines multiple json files into one file

import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('-i')
parser.add_argument('-o')
args = parser.parse_args()

filenames = os.listdir(args.i)

videos = {}
for filename in filenames:
    video = filename.split("_")[0]
    with open(os.path.join(os.getcwd(), args.i, filename), "r") as f:
        data = json.load(f)
        if video not in videos:
            videos[video] = [data]
        else:
            videos[video].append(data)

for video in videos:
    with open(os.path.join(os.getcwd(), args.o, video + ".json"), "w") as f:
        json.dump(videos[video], f)
