# vrep virtual deep learning

import vrep_commucation.vrep as vrep
import time
import numpy as np
import cv2
import csv
import random
import math
from math import cos ,sin

# A = np.array([[-15, 15, 1], [15, 15, 1], [15, -15, 1], [-15, -15, 1]])
# B = np.array([[0, 0, 1], [512, 0, 1], [512, 512, 1], [0, 512, 1]])
transforms = np.array([[1706.666667, 0, 0], [0.0, -1706.666667, 0], [256, 256, 1]])
path = "./training"
didct = {"image_name", "nuts", "screw"}
big_tmp = []

class Base:
    
    def __init__(self, filename, width, height, xmin, ymin, xmax, ymax):
        self.filename = filename
        self.width = width
        self.height = height
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
    
    def revack_limit():
        return self.xmin, self.ymin, self.xmax, self.ymax

class Nuts(Base):
    pass

class Screw(Base):
    pass

def in512(num):
    if num > 512:
        return 512
    if num < 0:
        return 0
    return num

def clasfier():
    image = cv2.imread("all.png")
    nut, screw = object_dectected(image)
    # (350, -320, 116, 0, 180, 0) left corner
    """
    pix2mm = 0.3968
    orignal_pos = (350, -320, 116)
    nuttmp = []
    for i, val in enumerate(nut):
        if i == 3:
            break
        cx = (nut[i + 1][0] - val[0]) * pix2mm
        cy = (nut[i + 1][1] - val[1]) * pix2mm
        this_point_pos = (orignal_pos[0] + cx, orignal_pos[1] + cy, orignal_pos[2])
        nuttmp.append(this_point_pos)
    print(nuttmp)
    """
    print(nut)
    print(screw)
    return nut, screw
    
def saveimg(image, image_name):
    img = np.array(image, dtype=np.uint8)
    img.resize([resolution[0], resolution[1], 3])
    img = np.rot90(img, 2)
    img = np.fliplr(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    file = f"{path}/{image_name}"
    print(file)
    cv2.imwrite(file, img)

vrep.simxFinish(-1)
while True:
    clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
    if clientID > -1:
        break
    else:
        time.sleep(0.2)
        print("Failed connecting to remote API server!")
print("Connection success!")

vrep.simxSynchronous(clientID, True)
name_list = ["nuts_1", "nuts_2", "nuts_3", "nuts_4", "screw_1", "screw_2", "screw_3", "screw_4"]
handles = []
for name in name_list:
    _, handle = vrep.simxGetObjectHandle(clientID, name, vrep.simx_opmode_blocking)
    handles.append(handle)
print(handles)

csvfile = open('training/output.csv', 'w', newline='')
writer = csv.writer(csvfile)
writer.writerow(['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])

for i in range(2):
    nuts_tmp = []
    screw_tmp = []
    # select position random
    testpos = [0.0, 0.0, 0.100000]
    for handle in handles:
        x = random.uniform(-14., 14.) / 100
        y = random.uniform(-14., 14.) / 100
        vrep.simxSetObjectPosition(clientID, handle, -1, [x, y, 0.10], vrep.simx_opmode_blocking)

    transangle = []
    for handle in handles:
        random_radion = random.uniform(-np.pi / 2, np.pi / 2)
        transangle.append(random_radion)
        vrep.simxSetObjectOrientation(clientID, handle, -1, (0, 0, random_radion), vrep.simx_opmode_blocking)

    _, camhandle = vrep.simxGetObjectHandle(clientID, 'Vision_sensor0', vrep.simx_opmode_blocking)
    _, resolution, image = vrep.simxGetVisionSensorImage(clientID, camhandle, 0, vrep.simx_opmode_streaming)

    time.sleep(0.5)
    _, resolution, image = vrep.simxGetVisionSensorImage(clientID, camhandle, 0, vrep.simx_opmode_buffer)
    time.sleep(0.5)
    saveimg(image, f"{i}.png")

    iter = 0
    ttpm = []
    for handle in handles:
        # get position
        _, pos = vrep.simxGetObjectPosition(clientID, handle, -1, vrep.simx_opmode_blocking)
        _, angle = vrep.simxGetObjectOrientation(clientID, handle, -1, vrep.simx_opmode_blocking)
        # x_axis
        _, min_x = vrep.simxGetObjectFloatParameter(clientID, handle, 15, vrep.simx_opmode_blocking) # 15
        _, max_x = vrep.simxGetObjectFloatParameter(clientID, handle, 18, vrep.simx_opmode_blocking) # 18
        # Y_axis
        _, min_y = vrep.simxGetObjectFloatParameter(clientID, handle, 16, vrep.simx_opmode_blocking) # 16
        _, max_y = vrep.simxGetObjectFloatParameter(clientID, handle, 19, vrep.simx_opmode_blocking) # 19
        
        angle[2] = angle[2] * -1
        p1 = np.array([max_x , max_y , 1])
        p3 = np.array([min_x , max_y , 1])
        p2 = np.array([min_x , min_y , 1])
        p4 = np.array([max_x , min_y , 1])
        
        transpos = np.array([[cos(angle[2]), -sin(angle[2]), 0], [sin(angle[2]), cos(angle[2]), 0], [0, 0, 1]])
        
        p1 = np.dot(p1, transpos)
        p3 = np.dot(p3, transpos)
        p2 = np.dot(p2, transpos)
        p4 = np.dot(p4, transpos)
        
        Mintmparray = np.array([p3[0] + pos[0], p1[1] + pos[1], 1])
        Maxtmparray = np.array([p4[0] + pos[0], p2[1] + pos[1], 1])
        
        Mintmparray = np.dot(Mintmparray, transforms)
        Maxtmparray = np.dot(Maxtmparray, transforms)
        
        p1 = np.dot([p1[0] + pos[0], p1[1] + pos[1], 1], transforms)
        p3 = np.dot([p3[0] + pos[0], p3[1] + pos[1], 1], transforms)
        p2 = np.dot([p2[0] + pos[0], p2[1] + pos[1], 1], transforms)
        p4 = np.dot([p4[0] + pos[0], p4[1] + pos[1], 1], transforms)
        
        maxXx = in512(int(round(max(p1[0], p3[0], p2[0], p4[0]), 0)))
        miniXx = in512(int(round(min(p1[0], p3[0], p2[0], p4[0]))))
        maxYy = in512(int(round(max(p1[1], p3[1], p2[1], p4[1]))))
        miniYy = in512(int(round(min(p1[1], p3[1], p2[1], p4[1]))))
        
        image = cv2.imread(f"{path}/{i}.png")
        ttpm.append([(miniXx, miniYy), (maxXx, maxYy)])
        cv2.rectangle(image, (miniXx, miniYy), (maxXx, maxYy), (0, 255, 0), 2)
        if iter < 4:  
            writer.writerow([f"{i}.png", 512, 512, "nuts", miniXx, miniYy, maxXx, maxYy])
            # nuts_tmp.append(Nuts(, 512, 512, miniXx, miniYy, maxXx, maxYy))
        else:
            # screw_tmp.append(Screw(f"{i}.png", 512, 512, miniXx, miniYy, maxXx, maxYy))
            writer.writerow([f"{i}.png", 512, 512, "screw", miniXx, miniYy, maxXx, maxYy])
        """
        cv2.line(image, (int(p1[0]), int(p1[1])), (int(p3[0]), int(p3[1])), (0, 250, 0), 2)
        cv2.line(image, (int(p3[0]), int(p3[1])), (int(p2[0]), int(p2[1])), (0, 250, 0), 2)
        cv2.line(image, (int(p2[0]), int(p2[1])), (int(p4[0]), int(p4[1])), (0, 250, 0), 2)
        cv2.line(image, (int(p4[0]), int(p4[1])), (int(p1[0]), int(p1[1])), (0, 250, 0), 2)
        """
        iter = iter + 1
        
    for i in range(8):
        cv2.rectangle(image, ttpm[i][0], ttpm[i][1], (0, 255, 0), 2)
    cv2.imshow("result", image)
    cv2.waitKey(0)
    # big_tmp.append([f"{i}.png", nuts_tmp, screw_tmp])

print(big_tmp)
csvfile.close()
exit()
# test for WRITE need to one to write
with open('training/output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # 寫入一列資料
    writer.writerow(['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
    for nadata in big_tmp:
        writer.writerow([nadata[0], 512, 512, ])
    
    for nut in nut_points:
        writer.writerow(["test", 512, 512, "nuts", nut[0], nut[1], (nut[0] + nut[2]), (nut[1] + nut[3])])
       
    for screw in screw_points:
        writer.writerow(["test", 512, 512, "screw", screw[0], screw[1], (screw[0] + screw[2]), (screw[1] + screw[3])])

exit()
