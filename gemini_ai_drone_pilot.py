import sys

import cv2
import numpy as np
import time
import os
import json
import os
import cv2
import numpy as np
import time
from dronekit import connect, VehicleMode, LocationGlobalRelative, LocationGlobal, Vehicle
import time
from pymavlink import mavutil
import math
import vertexai
import json
import google.generativeai as genai
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from typing import List, Dict, Tuple, Any
import vertexai
import json
import google.generativeai as genai
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor

CONFIG_PATH = "config.json"
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

# Initialize vertex ai
os.environ["GOOGLE_API_KEY"] = config.get("GOOGLE_API_KEY")
genai.configure(api_key=config.get("GOOGLE_API_KEY"))
vertexai.init(
    project=config.get('PROJECT_ID'),
    location=config.get('LOCATION')
)
# Initialize llm
multimodal_llm = ChatGoogleGenerativeAI(
    model=config.get("LLM")
)

vehicle = None
# Initialize to the vehicle
# connection_string: str = config.get("DRONE_CONNECTION_STRING")
# baud_rate: int = 57600
# vehicle: Vehicle
# try:
#     vehicle: Vehicle = connect(
#         connection_string,
#         baud=baud_rate,
#         wait_ready=True
#     ) # vehicle = connect('127.0.0.1:14550', wait_ready=True)
# except:
#     print("Drone Connection Problem")
#     sys.exit(0)

def arm_vehicle():
    """
    Arms the vehicle and waits until the vehicle is armed.
    """
    while not vehicle.is_armable:
        print("Waiting for vehicle to become armable...")
        time.sleep(1)

    print("Arming motors")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        print("Waiting for arming...")
        time.sleep(1)
    return "Vehicle is armed."

def move_drone_by_control(coltrol_signals:list[str]):
    for e in coltrol_signals:
        if "stay" in e:
            print(e)
        elif "up" in e.strip():
            print("DRONE - up(5cm)")
             # change_altitude(5)
        elif "down" in e.strip():
            print("DRONE - down(5cm)")
             # change_altitude(-5)
        elif "right" in e.strip():
            print("DRONE - right(5cm)")
             # move_drone_right_left(5)
        elif "straight" in e.strip():
            print("DRONE - straight(5)")
             # move_drone_forward_backward(5)
        elif "back" in e.strip():
            print("DRONE - back(5cm)")
            # move_drone_forward_backward(-5)
        elif "left" in e.strip():
            print("DRONE - left(5cm)")
            # move_drone_right_left(-5)


def move_drone_forward_backward(dNorth):
    """
    Move the drone forward (north) or backward (south) by `dNorth` meters.
    Positive `dNorth` moves the drone forward (north).
    Negative `dNorth` moves the drone backward (south).
    """
    currentLocation = vehicle.location.global_relative_frame
    targetLocation = get_location_metres(currentLocation, dNorth, 0)  # Move along the north-south axis
    targetDistance = get_distance_metres(currentLocation, targetLocation)

    print(f"Moving drone to target location (North/South): {targetLocation}, Distance: {targetDistance} meters")

    # Send the command to move the drone
    vehicle.simple_goto(targetLocation)

    # Monitor the movement and stop when it reaches the target location
    while vehicle.mode.name == "GUIDED":  # Stay in GUIDED mode
        remainingDistance = get_distance_metres(vehicle.location.global_relative_frame, targetLocation)
        print(f"Distance to target: {remainingDistance}")

        # Break the loop when within 1 meter of the target location
        if remainingDistance <= 1.0:
            print("Reached target location")
            break
        time.sleep(1)


def takeoff(altitude:int):
    """
    Initiates takeoff to a specified altitude in int
    :param altitude: altitude in int
    :return: vehicle's final altitude
    """
    print("Taking off!")
    vehicle.simple_takeoff(altitude)

    while True:
        print("Altitude: ", vehicle.location.global_relative_frame.alt)
        if vehicle.location.global_relative_frame.alt >= altitude * 0.95:
            print("Reached target altitude")
            break
        time.sleep(1)
    return vehicle.location.global_relative_frame.alt


def move_drone_right_left(dEast):
    """
    Move the drone right (east) or left (west) by `dEast` meters.
    Positive `dEast` moves the drone right (east).
    Negative `dEast` moves the drone left (west).
    """
    currentLocation = vehicle.location.global_relative_frame
    targetLocation = get_location_metres(currentLocation, 0, dEast)  # Move along the east-west axis (0 for north)
    targetDistance = get_distance_metres(currentLocation, targetLocation)

    print(f"Moving drone to target location (East/West): {targetLocation}, Distance: {targetDistance} meters")

    # Send the command to move the drone
    vehicle.simple_goto(targetLocation)

    # Monitor the movement and stop when it reaches the target location
    while vehicle.mode.name == "GUIDED":  # Stay in GUIDED mode
        remainingDistance = get_distance_metres(vehicle.location.global_relative_frame, targetLocation)
        print(f"Distance to target: {remainingDistance}")

        # Break the loop when within 1 meter of the target location
        if remainingDistance <= 1.0:
            print("Reached target location")
            break
        time.sleep(1)

def land():
    """
    Lands the vehicle.
    returns true or false for vehicle on state
    """
    print("Landing")
    vehicle.mode = VehicleMode("LAND")
    while vehicle.armed:
        print("Waiting for landing...")
        time.sleep(1)
    return vehicle.armed
#
# def goto(dNorth:int, dEast:int, gotoFunction=vehicle.simple_goto):
#     """
#     Moves the vehicle to a position dNorth metres North and dEast metres East of the current position.
#     """
#     currentLocation = vehicle.location.global_relative_frame
#     targetLocation = get_location_metres(currentLocation, dNorth, dEast)
#     targetDistance = get_distance_metres(currentLocation, targetLocation)
#     gotoFunction(targetLocation)


def get_distance_metres(aLocation1, aLocation2):
    """
    Returns the ground distance in metres between two LocationGlobal objects.
    """
    dlat = aLocation2.lat - aLocation1.lat
    dlong = aLocation2.lon - aLocation1.lon
    return math.sqrt((dlat * dlat) + (dlong * dlong)) * 1.113195e5


def get_location_metres(original_location, dNorth, dEast):
    """
    Returns a LocationGlobal object containing the latitude/longitude `dNorth` and `dEast` metres from the
    specified `original_location`. The returned LocationGlobal has the same `alt` value
    as `original_location`.

    :param original_location:
    :param dNorth:
    :param dEast:
    :return:
    """
    earth_radius = 6378137.0  # Radius of "spherical" earth
    # Coordinate offsets in radians
    dLat = dNorth / earth_radius
    dLon = dEast / (earth_radius * math.cos(math.pi * original_location.lat / 180))

    # New position in decimal degrees
    newlat = original_location.lat + (dLat * 180 / math.pi)
    newlon = original_location.lon + (dLon * 180 / math.pi)
    if type(original_location) is LocationGlobal:
        targetlocation = LocationGlobal(newlat, newlon, original_location.alt)
    elif type(original_location) is LocationGlobalRelative:
        targetlocation = LocationGlobalRelative(newlat, newlon, original_location.alt)
    else:
        raise Exception("Invalid Location object passed")

    return targetlocation


def change_altitude(target_altitude:int):
    """

    :param target_altitude:
    :return:
    """
    print("Changing altitude to:", target_altitude)
    current_location = vehicle.location.global_relative_frame
    target_location = LocationGlobalRelative(current_location.lat, current_location.lon, target_altitude)
    vehicle.simple_goto(target_location)

    while True:
        print("Altitude: ", vehicle.location.global_relative_frame.alt)
        if abs(vehicle.location.global_relative_frame.alt - target_altitude) < 0.5:
            print("Reached target altitude")
            break
        time.sleep(1)


def main():
    LAND  = False

    # arm_vehicle()

    # takeoff(10)

    # Create a data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Open the video capture device (adjust the index if needed)
    cap = cv2.VideoCapture(0)  # 0 is typically the default webcam; adjust if necessary

    if not cap.isOpened():
        print("Error: Couldn't open video stream.")
        return

    last_update_time = time.time()
    frame_skip = 2  # Process every 2nd frame
    frame_count = 0

    img_output_dir = "./imgs"
    if not os.path.exists(img_output_dir):
        os.makedirs(img_output_dir)
    data_output_dir = "./imgs_data"
    if not os.path.exists(data_output_dir):
        os.makedirs(data_output_dir)

    time_window = 0
    time_laps = 5
    while True:
        ret, frame = cap.read()
        timestamp = int(time.time())

        if time_window % time_laps == 0:
            image_path = os.path.join(img_output_dir, f"cam_{timestamp}.jpeg")
            cv2.imwrite(image_path, frame)
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "name the objects and find estimated distances from camera frame in cm."
                                "you always give your output in a dictionary"
                    },
                    {
                        "type": "image_url",
                        "image_url": image_path,
                    }
                ]
            )
            response = multimodal_llm.invoke(
                [message]
            ).content
            response_json = eval(response.replace("json", "").replace("```", ""))
            print("Object and Distances ===>", response_json)
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "You are a gamer, who predicts stay, up, straight, back, down, right and left moves of a toy drone."
                                "Given the object and their distances frm drone camera, provide a maneuver using only - "
                                " - stay, "
                                " - up(5cm),"
                                " - down(5cm), "
                                " - right(5cm), "
                                " - straight(5), "
                                " - back(5)"
                                " - left(5cm)"
                                "If any object's distance is less than 15 cm follow apply combination of maneuvers to avoid a collision."
                                f"Generate maneuver in a single line. Only generate given maneuvers, no extra lines please."
                                f" You can consider following as example:"
                                "INPUT: {'person': {'distance': 75}, 'fan': {'distance': 200}, 'window': {'distance': 250}}"
                                "OUTPUT:straight(5)"
                                "INPUT: {'person': {'distance': 15}, 'fan': {'distance': 200}, 'window': {'distance': 250}}"
                                "OUTPUT: stay up(5) left(5) straight(5) right(5) straight(5)"
                                f"INPUT:{response_json}"
                                f"OUTPUT:"
                    }
                ]
            )
            pilot_response = multimodal_llm.invoke(
                [message]
            ).content
            print("maneuver ===>", pilot_response)
            controls = pilot_response.split(" ")

            move_drone_by_control(coltrol_signals=controls)


        time_window += 1
        if not ret:
            print("Failed to grab frame")
            break

        if time_window > 300:
            LAND = True
            break


        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

    cap.release()
    cv2.destroyAllWindows()

    if LAND:
        land()

if __name__ == "__main__":
    main()
