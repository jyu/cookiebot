import os
undo = open('log.txt')
for f in undo:
  cmd = 'rm ' + f.replace('\n', '')
  os.system(cmd)
