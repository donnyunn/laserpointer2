shotDetected = False
endDetected = False
game_num = 0
pointArray = [(0,0), (0,0), (0,0)]

def recordResult(filename, game_num, pts1, pts2, pts3):
    f = open(filename, 'a')
    line = str(game_num) + ","
    line = line + str(pts1[0]) + ","
    line = line + str(pts1[1]) + ","
    line = line + str(pts2[0]) + ","
    line = line + str(pts2[1]) + ","
    line = line + str(pts3[0]) + ","
    line = line + str(pts3[1]) + ",\n"
    print(line)
    f.write(line)
    f.close()

def areDatumChanged(pts1, pts2, pts3):
    global pointArray
    ret = False

    if abs(pointArray[0][0] - pts1[0]) > 1 or abs(pointArray[0][1] - pts1[1]) > 1:
        ret = True
    elif abs(pointArray[1][0] - pts2[0]) > 1 or abs(pointArray[1][1] - pts2[1]) > 1:
        ret = True
    elif abs(pointArray[2][0] - pts3[0]) > 1 or abs(pointArray[2][1] - pts3[1]) > 1:
        ret = True
    pointArray[0] = pts1
    pointArray[1] = pts2
    pointArray[2] = pts3
    return ret

def areDatumVeryChanged(pts1, pts2, pts3):
    global pointArray
    ret = False

    if abs(pointArray[0][0] - pts1[0]) > 10 or abs(pointArray[0][1] - pts1[1]) > 10:
        ret = True
    elif abs(pointArray[1][0] - pts2[0]) > 10 or abs(pointArray[1][1] - pts2[1]) > 10:
        ret = True
    elif abs(pointArray[2][0] - pts3[0]) > 10 or abs(pointArray[2][1] - pts3[1]) > 10:
        ret = True
    pointArray[0] = pts1
    pointArray[1] = pts2
    pointArray[2] = pts3
    return ret

def billiardRule(filename, pts1, pts2, pts3):
    global shotDetected, endDetected, game_num
    if shotDetected:
        if not areDatumChanged(pts1, pts2, pts3):
            print("end detected")
            shotDetected = False
            endDetected = True
    else:
        if endDetected:
            recordResult(filename, game_num, pts1, pts2, pts3)
            game_num = game_num + 1
            endDetected = False
        else:
            if areDatumVeryChanged(pts1, pts2, pts3):
                print("shot detected")
                shotDetected = True