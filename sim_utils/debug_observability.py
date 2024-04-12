from utils import point_in_trapezoid
import numpy as np 
import matplotlib.pyplot as plt 

def get_frustrum_verts(robot_pose,max_range=100): 
    min_d = 0.1; max_d = max_range*1.1
    if len(robot_pose) == 6:
        min_theta = np.deg2rad(robot_pose[5] - (72/2)) - np.deg2rad(2)
        max_theta = np.deg2rad(robot_pose[5] + (72/2)) + np.deg2rad(2)
        if robot_pose[5] > 2*np.pi:
            raise OSError 
    elif len(robot_pose) == 3:
        min_theta = np.deg2rad(robot_pose[2] - (72/2)) - np.deg2rad(2) 
        max_theta = np.deg2rad(robot_pose[2] + (72/2)) + np.deg2rad(2)
        if robot_pose[2] > 2*np.pi:
            raise OSError
    x0 = robot_pose[0] + min_d*np.cos(min_theta)
    y0 = robot_pose[1] + min_d*np.sin(min_theta)
    x1 = robot_pose[0] + min_d*np.cos(max_theta)
    y1 = robot_pose[1] + min_d*np.sin(max_theta)
    x2 = robot_pose[0] + max_d*np.cos(min_theta)
    y2 = robot_pose[1] + max_d*np.sin(min_theta)
    x3 = robot_pose[0] + max_d*np.cos(max_theta)
    y3 = robot_pose[1] + max_d*np.sin(max_theta)
    return [(x0,y0),(x1,y1),(x2,y2),(x3,y3)]

car_pose = np.array([397.31761298, 318.05003192,  -1.56706369])

world_pt = np.array([392.34856682, 132.85033974,  0.91688012])
verts = [(396.25281276959106, 232.1858729143586), (396.37594506293556, 232.18585232970048), (328.57713981186123, 145.5948035727781), (464.0226624908224, 145.5721604488458)]

max_range = 100 
sixd_cam_pose = np.array([396.6251525878906, 317.58734130859375, 1.701406717300415, 0.0015258764954767256, -0.11378410279254789, -90.38040306326697])

if point_in_trapezoid(world_pt,verts):
    print("Point in Trap!")
else:
    print("Point outside trap :(")
    #if np.linalg.norm(world_pt - sixd_cam_pose[:3]) < max_range:
    #plot the car with points

fig,ax = plt.subplots()
#ax.scatter(car_pose[0],car_pose[1],color="k")
'''
if len(car_pose) == 6:
    yaw = car_pose[5]
    if np.abs(yaw) > 2*np.pi:
        yaw = np.deg2rad(yaw)
elif len(car_pose) == 3:
    yaw = car_pose[2]
    if np.abs(yaw) > 2*np.pi:
        yaw = np.deg2rad(yaw)
pointer_x = car_pose[0] + 2*np.cos(yaw)
pointer_y = car_pose[1] + 2*np.sin(yaw)
'''
#ax.plot([car_pose[0],pointer_x],[car_pose[1],pointer_y],"k")
#verts = get_frustrum_verts(sixd_cam_pose)
x0,y0 = verts[0]
x1,y1 = verts[1]
x2,y2 = verts[2]
x3,y3 = verts[3]
ax.plot([x0,x1],[y0,y1],'b')
ax.plot([x1,x2],[y1,y2],'b')
ax.plot([x2,x3],[y2,y3],'b')
ax.plot([x3,x0],[y3,y0],'b')
ax.scatter(world_pt[0],world_pt[1],color="r",marker="*")
ax.set_aspect('equal')
plt.show(block=True)