#!/usr/bin/env python3
# Ganapathy Hari Narayan
# Pranathi Tupakula
''' Get a raw frame from camera and display in OpenCV
By press space, save the image from 001.bmp to ...
'''

import cv2
import cozmo
import numpy as np
import asyncio
import sys
from numpy.linalg import inv
import threading
from queue import PriorityQueue

from ar_markers.hamming.detect import detect_markers
from grid import *
from visualizer import *
from gui import GUIWindow
from particle import Particle, Robot
from setting import *
from particle_filter import *
from utils import *
import cv2
import numpy as np
import find_ball
from cozmo.util import degrees, radians, distance_inches, speed_mmps
from cozmo import anim
import time


# camera params
camK = np.matrix([[295, 0, 160], [0, 295, 120], [0, 0, 1]], dtype='float32')

#marker size in inches
marker_size = 3.5

# tmp cache
last_pose = cozmo.util.Pose(0,0,0,angle_z=cozmo.util.Angle(degrees=0))
flag_odom_init = False

# goal location for the robot to drive to, (x, y, theta)
goal = (6,10,0)

# map
Map_filename = "map_arena.json"
grid = CozGrid(Map_filename)
gui = GUIWindow(grid)

BALL_SIZE = 4.572
image_size = 34.546*2
ball_dist= 15.24
FOCAL_LENGTH = 132.7;



async def image_processing(robot):

    global camK, marker_size

    event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)

    # convert camera image to opencv format
    opencv_image = np.asarray(event.image)

    # detect markers
    markers = detect_markers(opencv_image, marker_size, camK)

    # show markers
    for marker in markers:
        marker.highlite_marker(opencv_image, draw_frame=True, camK=camK)
        #print("ID =", marker.id);
        #print(marker.contours);
    cv2.imshow("Markers", opencv_image)

    return markers


def astar(grid, heuristic):
    """Perform the A* search algorithm on a defined grid

        Arguments:
        grid -- CozGrid instance to perform search on
        heuristic -- supplied heuristic function
    """

    # Your code here
    start = grid.getStart()
    goal = grid.getGoals()
    grid.clearVisited()

    current = start
    print(current)
    pq = PriorityQueue()
    pq.put((heuristic(current, goal) + 0, (start, 0, [current])))

    while not pq.empty():
        node = pq.get()
        current = node[1][0]
        visited = grid.getVisited()

        if current in visited:
            continue

        if current == goal[0]:
            print("DONE")
            grid.setPath(node[1][2])
            break

        oldDist = node[1][1]
        oldPath = node[1][2]

        grid.addVisited(current)
        nodeNeighbours = grid.getNeighbors(current)

        for n in nodeNeighbours:
            if n in visited:
                continue
            else:
                pq.put((heuristic(n[0], goal) + oldDist + n[1], (n[0], oldDist + n[1], oldPath + [n[0]])))

def heuristic(current, goal):
    """Heuristic function for A* algorithm

        Arguments:
        current -- current cell
        goal -- desired goal cell
    """

    return ((goal[0][0] - current[0])**2 + (goal[0][1] - current[1])**2) ** (0.5) # Your code here
#calculate marker pose
def cvt_2Dmarker_measurements(ar_markers):

    marker2d_list = [];

    for m in ar_markers:
        R_1_2, J = cv2.Rodrigues(m.rvec)
        R_1_1p = np.matrix([[0,0,1], [0,-1,0], [1,0,0]])
        R_2_2p = np.matrix([[0,-1,0], [0,0,-1], [1,0,0]])
        R_2p_1p = np.matmul(np.matmul(inv(R_2_2p), inv(R_1_2)), R_1_1p)
        #print('\n', R_2p_1p)
        yaw = -math.atan2(R_2p_1p[2,0], R_2p_1p[0,0])

        x, y = m.tvec[2][0] + 0.5, -m.tvec[0][0]
        # print('x =', x, 'y =', y,'theta =', yaw)

        # remove any duplate markers
        dup_thresh = 2.0
        find_dup = False
        for m2d in marker2d_list:
            if grid_distance(m2d[0], m2d[1], x, y) < dup_thresh:
                find_dup = True
                break
        if not find_dup:
            marker2d_list.append((x,y,math.degrees(yaw)))

    return marker2d_list


#compute robot odometry based on past and current pose
def compute_odometry(curr_pose, cvt_inch=True):
    global last_pose, flag_odom_init
    last_x, last_y, last_h = last_pose.position.x, last_pose.position.y, \
        last_pose.rotation.angle_z.degrees
    curr_x, curr_y, curr_h = curr_pose.position.x, curr_pose.position.y, \
        curr_pose.rotation.angle_z.degrees

    dx, dy = rotate_point(curr_x-last_x, curr_y-last_y, -last_h)
    if cvt_inch:
        dx, dy = dx / 25.6, dy / 25.6

    return (dx, dy, diff_heading_deg(curr_h, last_h))

#particle filter functionality
class ParticleFilter:

    def __init__(self, grid):
        self.particles = Particle.create_random(PARTICLE_COUNT, grid)
        self.grid = grid

    def update(self, odom, r_marker_list):

        # ---------- Motion model update ----------
        self.particles = motion_update(self.particles, odom)

        # ---------- Sensor (markers) model update ----------
        self.particles = measurement_update(self.particles, r_marker_list, self.grid)

        # ---------- Show current state ----------
        # Try to find current best estimate for display
        m_x, m_y, m_h, m_confident = compute_mean_pose(self.particles)
        return (m_x, m_y, m_h, m_confident)

async def run(robot: cozmo.robot.Robot):

    global flag_odom_init, last_pose
    global grid, gui

    # start streaming
    robot.camera.image_stream_enabled = True

    #start particle filter
    pf = ParticleFilter(grid)
    pickup = True
    counter = 0;
    await robot.set_head_angle(degrees(5)).wait_for_completed();

    ###################
    t0 = 0
    t1 = 0
    state = 0
    count = 5
    await robot.set_lift_height(0.0, accel=10.0, max_speed=10.0, duration=0.0, in_parallel=False, num_retries=0).wait_for_completed()
    robot.stop_all_motors()
    #await robot.set_lift_height(0.5, accel=20.0, max_speed=20.0, duration=0.0, in_parallel=False, num_retries=0).wait_for_completed()
    ############YOUR CODE HERE#################
    print(robot.pose.position)
    while True:
        curr_pose = robot.pose
        diff = compute_odometry(curr_pose)
        ar_markers_list = await image_processing(robot)
        marker_pose_list = cvt_2Dmarker_measurements(ar_markers_list)

        estimated = pf.update(diff, marker_pose_list)
        flag = estimated[3]

        gui.show_particles(pf.particles)
        gui.show_mean(estimated[0], estimated[1], estimated[2], estimated[3])
        gui.updated.set()

        # print(robot.pose.rotation.angle_z.degrees)

        t0 = time.time()
        
        if state == 0:
            #await robot.set_lift_height(0.5, accel=20.0, max_speed=20.0, duration=0.0, in_parallel=False, num_retries=0).wait_for_completed()
            #await robot.set_head_angle(degrees(-3)).wait_for_completed()
            offset = 0
            speed = 30
            while (t1 - t0) < (count):
                event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)

                #convert camera image to opencv format
                opencv_image = cv2.cvtColor(np.asarray(event.image), cv2.COLOR_RGB2GRAY)
                ball = find_ball.find_ball(opencv_image)
                print(ball)
                
                #find the ball
                if ball is not None:
                    x = len(opencv_image[0])/2
                    offset = x - ball[0]
                    print(offset)
                await robot.drive_straight(distance_inches(0), speed=speed_mmps(500)).wait_for_completed()
                if (offset > 0):
                    await robot.drive_wheels(speed, speed + offset/20, duration=0)
                else:
                    await robot.drive_wheels(speed + (-offset/20), speed, duration=0)
                #await robot.set_lift_height(0.0, accel=20.0, max_speed=20.0, duration=0.0, in_parallel=False, num_retries=0).wait_for_completed()
                t1 = time.time()
                speed = speed + 4
            robot.stop_all_motors()
            await robot.drive_wheels(-27, -27, duration=4)
            robot.stop_all_motors()
            state = 1
            await robot.set_lift_height(0.0, accel=20.0, max_speed=20.0, duration=0.0, in_parallel=False, num_retries=0).wait_for_completed()
        if state == 1:
            await robot.set_lift_height(0.0, accel=20.0, max_speed=20.0, duration=0.0, in_parallel=False, num_retries=0).wait_for_completed()
            if not flag:
                await robot.drive_wheels(-10, 10, duration=0)
                await robot.set_head_angle(degrees(5)).wait_for_completed()
            else:
                robot.stop_all_motors()
                state = 2
                robot_curr_pos = (estimated[0], estimated[1], estimated[2])
                print(robot_curr_pos)
                print(robot.pose.position)
                grid.setStart((math.ceil(estimated[0]), math.ceil(estimated[1])))
        if state == 2:
            await robot.set_head_angle(degrees(0)).wait_for_completed();
            offset = 5000
            print("HELLO WUT")
            while True:
                #get camera image
                event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)

                #convert camera image to opencv format
                opencv_image = cv2.cvtColor(np.asarray(event.image), cv2.COLOR_RGB2GRAY)
                
                #find the ball
                ball = find_ball.find_ball(opencv_image)

                ##ENTER YOUR GOTO_BALL SOLUTION HERE
                
                if ball is None:
                    print("Ball is none")
                else:
                    x = len(opencv_image[0])/2
                    offset = math.fabs(x - ball[0])

                if ball is None or offset > 6:
                    #await robot.turn_in_place(degrees(45)).wait_for_completed()
                    await robot.drive_wheels(10, -10, duration=0)
                else:
                    print(offset)
                    state = 3
                    break
        if state == 3:
            robot.stop_all_motors()
            image_size = ball[2]
            ball_dist = FOCAL_LENGTH * 40 / image_size
            ball_dist = ball_dist / 25.4
            angle = robot.pose.rotation.angle_z.degrees
            print("angle is", angle)
            position = -1
            print(ball_dist)
            if angle < 0:
                angle = angle + 180
                if (angle < 15):
                    position = 3
                else:
                    position = 1
                print(angle, "less")
                print(ball_dist)
                print(math.sin(math.radians(angle)))
                x_coord = math.cos(math.radians(angle)) * ball_dist
                y_coord = math.sin(math.radians(angle)) * ball_dist
                ball_coord = (robot_curr_pos[0] - x_coord ,robot_curr_pos[1]- y_coord)
            else:
                angle = 180 - angle
                if angle < 15:
                    position = 3
                else:
                    position = 2
                print(angle, "greater")
                x_coord = math.cos(math.radians(angle)) * ball_dist
                y_coord = math.sin(math.radians(angle)) * ball_dist
                ball_coord = (robot_curr_pos[0] - x_coord, y_coord + robot_curr_pos[1])
            ball_coord = (math.floor(ball_coord[0]), math.floor(ball_coord[1]))
            final_dist = math.hypot(25 - ball_coord[0], ball_coord[1] - 14)
            count = (final_dist * 26/26)
            print(ball_coord)
            state = 4
        if state == 4:
            robot.stop_all_motors()
            if position == 1:
                dx = robot_curr_pos[0] - (ball_coord[0] - 6)
                if ball_dist < 7:
                    dy = robot_curr_pos[1] - ball_coord[1] + 3
                else:
                    dy = robot_curr_pos[1] - ball_coord[1] + 1
                await robot.turn_in_place(degrees(-angle), in_parallel=False, num_retries=0).wait_for_completed()
                robot.stop_all_motors()    
                await robot.drive_straight(distance_inches(dx), speed=speed_mmps(26)).wait_for_completed()
                await robot.turn_in_place(degrees(90), in_parallel=False, num_retries=0).wait_for_completed()
                robot.stop_all_motors()
                await robot.drive_straight(distance_inches(dy), speed=speed_mmps(26)).wait_for_completed()
                robot.stop_all_motors()
                while True:
                    #get camera image
                    event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)

                    #convert camera image to opencv format
                    opencv_image = cv2.cvtColor(np.asarray(event.image), cv2.COLOR_RGB2GRAY)
                    
                    #find the ball
                    ball = find_ball.find_ball(opencv_image)

                    ##ENTER YOUR GOTO_BALL SOLUTION HERE
                    
                    if ball is None:
                        print("Ball is none")
                    else:
                        x = len(opencv_image[0])/2
                        offset = math.fabs(x - ball[0])

                    if ball is None or offset > 6:
                        #await robot.turn_in_place(degrees(45)).wait_for_completed()
                        await robot.drive_wheels(-10, 10, duration=0)
                    else:
                        print(offset)
                        robot.stop_all_motors()
                        break
                offset = 0
                t0 = time.time()
                speed = 30
                while (t1 - t0) < (count):
                    # await robot.set_lift_height(0.0, accel=10.0, max_speed=10.0, duration=0.0, in_parallel=False, num_retries=0).wait_for_completed()
                    # await robot.set_lift_height(0.5, accel=10.0, max_speed=10.0, duration=0.0, in_parallel=False, num_retries=0).wait_for_completed()
                    event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)

                    #convert camera image to opencv format
                    opencv_image = cv2.cvtColor(np.asarray(event.image), cv2.COLOR_RGB2GRAY)
                    ball = find_ball.find_ball(opencv_image)
                    print(ball)
                    
                    #find the ball
                    if ball is not None:
                        x = len(opencv_image[0])/2
                        offset = x - ball[0]
                        print(offset)
                    await robot.drive_straight(distance_inches(0), speed=speed_mmps(500)).wait_for_completed()
                    if (offset > -20):
                        await robot.drive_wheels(speed, speed + offset/20, duration=0)
                    else:
                        await robot.drive_wheels(speed + (-offset/20), speed, duration=0)
                    #await robot.set_lift_height(0.0, accel=20.0, max_speed=20.0, duration=0.0, in_parallel=False, num_retries=0).wait_for_completed()
                    t1 = time.time()
                    speed = speed + 4
                state = 1
                robot.stop_all_motors()
            elif position == 2:
                dx = robot_curr_pos[0] - (ball_coord[0] - 6)
                if ball_dist < 7:
                    dy = ball_coord[1] - robot_curr_pos[1] + 1.5
                else:
                    dy = ball_coord[1] - robot_curr_pos[1] + 1
                await robot.turn_in_place(degrees(angle - 2), in_parallel=False, num_retries=0).wait_for_completed()
                robot.stop_all_motors()    
                await robot.drive_straight(distance_inches(dx), speed=speed_mmps(26)).wait_for_completed()
                await robot.turn_in_place(degrees(-90), in_parallel=False, num_retries=0).wait_for_completed()
                robot.stop_all_motors()
                await robot.drive_straight(distance_inches(dy), speed=speed_mmps(26)).wait_for_completed()
                robot.stop_all_motors()
                while True:
                    #get camera image
                    event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)

                    #convert camera image to opencv format
                    opencv_image = cv2.cvtColor(np.asarray(event.image), cv2.COLOR_RGB2GRAY)
                    
                    #find the ball
                    ball = find_ball.find_ball(opencv_image)

                    ##ENTER YOUR GOTO_BALL SOLUTION HERE
                    
                    if ball is None:
                        print("Ball is none")
                    else:
                        x = len(opencv_image[0])/2
                        offset = math.fabs(x - ball[0])

                    if ball is None or offset > 6:
                        #await robot.turn_in_place(degrees(45)).wait_for_completed()
                        await robot.drive_wheels(10, -10, duration=0)
                    else:
                        print(offset)
                        robot.stop_all_motors()
                        state = 0
                        break
                robot.stop_all_motors()
                state = 0
            else:
                await robot.turn_in_place(degrees(-90), in_parallel=False, num_retries=0).wait_for_completed()
                robot.stop_all_motors()
                await robot.drive_straight(distance_inches(4.5), speed=speed_mmps(26)).wait_for_completed()
                robot.stop_all_motors()
                await robot.turn_in_place(degrees(90), in_parallel=False, num_retries=0).wait_for_completed()
                robot.stop_all_motors()
                await robot.drive_straight(distance_inches(ball_dist + 7), speed=speed_mmps(26)).wait_for_completed()
                robot.stop_all_motors()
                await robot.turn_in_place(degrees(90), in_parallel=False, num_retries=0).wait_for_completed()
                robot.stop_all_motors()
                await robot.drive_straight(distance_inches(4.5), speed=speed_mmps(26)).wait_for_completed()
                robot.stop_all_motors()
                #await robot.turn_in_place(degrees(90), in_parallel=False, num_retries=0).wait_for_completed()
                
                while True:
                    #get camera image
                    event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)

                    #convert camera image to opencv format
                    opencv_image = cv2.cvtColor(np.asarray(event.image), cv2.COLOR_RGB2GRAY)
                    
                    #find the ball
                    ball = find_ball.find_ball(opencv_image)

                    ##ENTER YOUR GOTO_BALL SOLUTION HERE
                    
                    if ball is None:
                        print("Ball is none")
                    else:
                        x = len(opencv_image[0])/2
                        offset = math.fabs(x - ball[0])

                    if ball is None or offset > 6:
                        #await robot.turn_in_place(degrees(45)).wait_for_completed()
                        await robot.drive_wheels(10, -10, duration=0)
                    else:
                        print(offset)
                        robot.stop_all_motors()
                        state = 0
                        break
                robot.stop_all_motors()
                state = 0



            # print(state)
            # astar(grid, heuristic)
            # grid.addGoal((1, 1))
            # astar(grid, heuristic)
            # print(grid.getPath())
        last_pose = curr_pose


    ###################


class CozmoThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self, daemon=False)

    def run(self):
        cozmo.robot.Robot.drive_off_charger_on_connect = False  # Cozmo can stay on his charger
        cozmo.run_program(run, use_viewer=False)


if __name__ == '__main__':

    # cozmo thread
    cozmo_thread = CozmoThread()
    cozmo_thread.start()

    astar_grid = CozGrid("emptygrid.json")
    visualizer = Visualizer(grid)
    updater = UpdateThread(visualizer)
    updater.start()
    visualizer.start()
    # init
    grid = CozGrid(Map_filename)
    gui = GUIWindow(grid)
    gui.start()

