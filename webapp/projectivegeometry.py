import numpy as np
import math
import os

ini_folder_directory = os.path.dirname(os.path.abspath(__file__)) + '/ini_data/'

def Coefficients(input):
    p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, q1x, q1y, q2x, q2y, q3x, q3y, q4x, q4y = input
    # p : camera coordinate,
    # q : laser coordinate
    print(input)
    p1 = [p1x, p1y]
    p2 = [p2x, p2y]
    p3 = [p3x, p3y]
    p4 = [p4x, p4y]

    q1 = [q1x, q1y]
    q2 = [q2x, q2y]
    q3 = [q3x, q3y]
    q4 = [q4x, q4y]

    Mpq = np.array([ [p1[0], p1[1], 1, 0, 0, 0, -p1[0]*q1[0], -p1[1]*q1[0] ],
                     [0, 0, 0, p1[0], p1[1], 1, -p1[0]*q1[1], -p1[1]*q1[1] ],
                     [p2[0], p2[1], 1, 0, 0, 0, -p2[0]*q2[0], -p2[1]*q2[0] ],
                     [0, 0, 0, p2[0], p2[1], 1, -p2[0]*q2[1], -p2[1]*q2[1] ],
                     [p3[0], p3[1], 1, 0, 0, 0, -p3[0]*q3[0], -p3[1]*q3[0] ],
                     [0, 0, 0, p3[0], p3[1], 1, -p3[0]*q3[1], -p3[1]*q3[1] ],
                     [p4[0], p4[1], 1, 0, 0, 0, -p4[0]*q4[0], -p4[1]*q4[0] ],
                     [0, 0, 0, p4[0], p4[1], 1, -p4[0]*q4[1], -p4[1]*q4[1] ]])
    # print(Mpq)

    Mpq_inv = np.linalg.inv(Mpq)
    # print(Mpq_inv)

    Q = np.vstack([q1[0], q1[1], q2[0], q2[1], q3[0], q3[1], q4[0], q4[1]])
    # print(Q)

    X = np.dot(Mpq_inv, Q)
    # print(X)

    # Transmat = np.array([ [X[0], X[1], X[2] ], [ X[3], X[4], X[5] ], [ X[6], X[7], 1] ])
    # print(Transmat)

    A = X[0]
    B = X[1]
    C = X[2]
    D = X[3]
    E = X[4]
    F = X[5]
    G = X[6]
    H = X[7]
    return A, B, C, D, E, F, G, H

def getArray(n, cam, las, i, j , k, l):
    return np.array([cam[n][i][0]/10, cam[n][i][1]/10, \
                    cam[n][j][0]/10, cam[n][j][1]/10, \
                    cam[n][k][0]/10, cam[n][k][1]/10, \
                    cam[n][l][0]/10, cam[n][l][1]/10, \
                    las[n][i][0]/10, las[n][i][1]/10, \
                    las[n][j][0]/10, las[n][j][1]/10, \
                    las[n][k][0]/10, las[n][k][1]/10, \
                    las[n][l][0]/10, las[n][l][1]/10])

def getMatrix(laserNum, m):
    cam_n = []
    las_n = []
    for n in range(3):
        cam = []
        las = []
        filename = ini_folder_directory + "coord_list_%d.txt"%n
        f = open(filename, 'r')
        line = f.readline()
        xs = line.split(', ')
        line = f.readline()
        ys = line.split(', ')
        for i in range(len(xs) - 1):
            if i%2 is 0:
                cam.append([int(xs[i]), int(ys[i])])
            else:
                las.append([int(xs[i]), int(ys[i])])
        f.close()
        
        cam_n.append(cam)
        las_n.append(las)
    a = getArray(laserNum, cam_n, las_n, m, m-1, m+8, m+9)
    return Coefficients(a)

def getMatrix2(laserNum, m):
    cam_n = []
    las_n = []
    for n in range(3):
        cam = []
        las = []
        filename = ini_folder_directory + "coord_list_%d.txt"%n
        f = open(filename, 'r')
        line = f.readline()
        xs = line.split(', ')
        line = f.readline()
        ys = line.split(', ')
        for i in range(len(xs) - 1):
            if i%2 is 0:
                cam.append([int(xs[i]), int(ys[i])])
            else:
                las.append([int(xs[i]), int(ys[i])])
        f.close()
        
        cam_n.append(cam)
        las_n.append(las)
    a = getArray(laserNum, cam_n, las_n, m, m-1, m+4, m+5)
    return Coefficients(a)
    # print("\n\t    A = %f\n\t    B = %f\n\t    C = %f\n\t    D = %f\n\t    E = %f\n\t    F = %f\n\t    G = %f\n\t    H = %f\n"%(Coefficients(a)))
    