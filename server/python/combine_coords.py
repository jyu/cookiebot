#!/usr/bin/python3
# combines a file of x coords and a file of y coords into one file

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-x')
parser.add_argument('-y')
parser.add_argument('-c')
args = parser.parse_args()

with open(args.x, "r") as x_file:
    with open(args.y, "r") as y_file:
        with open(args.c, "w") as c_file:
            x_lines = x_file.readlines()
            y_lines = y_file.readlines()
            c_lines = []
            for i in range(len(x_lines)):
                c_lines.append(x_lines[i].rstrip() + " " +  y_lines[i].rstrip())
            c_file.write("\n".join(c_lines))
