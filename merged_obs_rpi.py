import cv2
import numpy as np
import time
import os
import json
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
import math

# Load the MobileNetSSD model for object detection
proto_path = 'MobileNetSSD_deploy.prototxt'
model_path = 'MobileNetSSD_deploy.caffemodel'
net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# Connect to the vehicle
connection_string = '/dev/ttyAMA0'
baud_rate = 57600
vehicle = connect(connection_string, baud=baud_rate, wait_ready=False)

# object avoidance variables
confidence_threshold = 0.85
frame_skip = 4  # Process every nth frame
last_update_time = time.time()


def detect_obstacles(frame, net, confidence_threshold):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    obstacles = []

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            obstacles.append((startX, startY, endX, endY))
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    return frame, obstacles


def avoidance_trajectory(obstacles):
    avoidance_command = None
    for (startX, startY, endX, endY) in obstacles:
        centerX = (startX + endX) // 2

        if centerX < 320:  # Left side
            avoidance_command = "Move Right"
        elif centerX > 640:  # Right side
            avoidance_command = "Move Left"
        else:
            avoidance_command = "Move Up"
    return avoidance_command


def arm_vehicle():
    while not vehicle.is_armable:
        print("Waiting for vehicle to become armable...")
        time.sleep(1)
    print("Arming motors")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True
    while not vehicle.armed:
        print("Waiting for arming...")
        time.sleep(1)

    print("Vehicle is armed.")


def takeoff(altitude):
    print("Taking off!")
    vehicle.simple_takeoff(altitude)
    while True:
        print("Altitude: ", vehicle.location.global_relative_frame.alt)
        if (vehicle.location.global_relative_frame.alt >= altitude * 0.95):
            print("Reached target altitude")
            break
        time.sleep(1)


def move_ned_velocity(velocity_x, velocity_y, velocity_z, duration):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,
        0, 0,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111000111,  # bitmask to indicate which dimensions should be ignored
        0, 0, 0,
        velocity_x, velocity_y, velocity_z,
        0, 0, 0,
        0, 0
    )
    for _ in range(duration):
        vehicle.send_mavlink(msg)
        time.sleep(1)


def land():
    print("Landing")
    vehicle.mode = VehicleMode("LAND")
    while vehicle.armed:
        print("Waiting for landing...")
        time.sleep(1)


def obstacle_avoidance_maneuver(command):
    if command == "Move Right":
        move_ned_velocity(1, 0, 0, 2)  # Move right
    elif command == "Move Left":
        move_ned_velocity(-1, 0, 0, 2)  # Move left
    elif command == "Move Up":
        move_ned_velocity(0, 0, -1, 2)  # Move upward


def goto_location(lat, lon, alt):
    """ Moves the vehicle to the specified latitude, longitude, and altitude. """
    target_location = LocationGlobalRelative(lat, lon, alt)
    print(f"Going to target location: Latitude: {lat}, Longitude: {lon}, Altitude: {alt}")
    vehicle.simple_goto(target_location)


def main():
    # Open the video capture device (adjust the index if needed)
    cap = cv2.VideoCapture(0)  # 0 is typically the default webcam; adjust if necessary

    if not cap.isOpened():
        print("Error: Couldn't open the video stream.")
        return

    arm_vehicle()
    takeoff(10)

    # Prompt for the target latitude and longitude
    try:
        lat = float(input("Enter target latitude: "))
        lon = float(input("Enter target longitude: "))
        alt = float(input("Enter target altitude: "))
    except ValueError:
        print("Invalid input. Please enter valid latitude, longitude, and altitude.")
        return

    # Go to the specified location
    goto_location(lat, lon, alt)

    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            # resizing frames to reduce processing time
            frame = cv2.resize(frame, (640, 480))

            frame_with_obstacles, obstacles = detect_obstacles(frame, net, confidence_threshold)

            # Refine the confidence threshold based on recent detections
            current_time = time.time()
            if current_time - last_update_time > 3:  # Update every three seconds
                last_update_time = current_time

            # check for obstacles and decide avoidance maneuver
            avoidance_command = avoidance_trajectory(obstacles)

            if avoidance_command:
                print(f"Avoiding obstacle: {avoidance_command}")
                obstacle_avoidance_maneuver(avoidance_command)

            # Display the frame
            cv2.imshow("Drone Live Stream", frame_with_obstacles)

            # press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        land()

    except Exception as e:
        print(f"Error occurred: {e}")
        print("Landing the vehicle...")
        land()

    finally:
        print("Closing the vehicle...")
        vehicle.armed = False
        vehicle.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
