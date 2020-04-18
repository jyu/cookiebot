#!/bin/bash

while getopts v: option
  do 
  case "${option}"
  in
  v) VIDEO=$OPTARG;;
  esac
  done

echo "Running on video $VIDEO"
scp $VIDEO cookie:~/cookiebot/xavier/data
ssh cookie "cd ~/cookiebot/xavier/; python camera.py -v ${VIDEO}";
scp cookie:~/cookiebot/xavier/out/${VIDEO} out/${VIDEO}
