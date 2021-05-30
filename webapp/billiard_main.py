import os.path
import recogballs
import time

folder_directory = os.path.dirname(os.path.abspath(__file__)) + '/resources/'
file_1_directory = folder_directory + '1'
while True:
    if os.path.isfile(file_1_directory) == True:
        print("Game Start!")
        recogballs.loop()
        break
    time.sleep(1)

print("Game Finish!")