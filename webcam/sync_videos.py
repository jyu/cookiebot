import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-dir', default="point_videos", help="Name of video dir") # Video dir
args = parser.parse_args()

v_dir = args.dir
remote_dir = 'cookiebot/xavier/' + v_dir

for c in os.listdir(v_dir):
  v_path = v_dir + '/' + c
  local_fs = os.listdir(v_path)

  remote_path = remote_dir + '/' + c
  remote_cmd = 'ssh cookie "ls ' + remote_path + '"'
  res = os.popen(remote_cmd).read().split('\n')
  res = list(filter(lambda x: '.mp4' in x, res))
  
  for l in local_fs:
    if l in res:
      continue

    cmd = 'scp ' + v_path + '/' + l + ' cookie:' + remote_path
    os.system(cmd)

