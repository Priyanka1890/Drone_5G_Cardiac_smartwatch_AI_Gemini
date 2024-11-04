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

CONFIG_PATH: str = "config.json"
MEMORY_KEY: str = "chat_history"
SYSTEM_PROMPT: str = ""
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)


# Initialize the MobileNetSSD model
proto_path: str = config.get('CAM_PROTO_PATH')
model_path: str = config.get('CAM_MODEL_PATH')
net: cv2.dnn.Net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
confidence_threshold: float = config.get("CAM_MOBILE_NET_SSD_CONFIDENCE_THRESHOLD")

# Initialize to the vehicle
connection_string: str = config.get("DRONE_CONNECTION_STRING")
baud_rate: int = 57600
vehicle: Vehicle
try:
    vehicle: Vehicle = connect(
        connection_string,
        baud=baud_rate,
        wait_ready=True
    ) # vehicle = connect('127.0.0.1:14550', wait_ready=True)
except:
    pass

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

# create tools as functions


@tool
def image_query(query:str, image_url:str):
    """
    generates answers from an image url and query in text
    :param query:str, query for the given image in plain text
    :param image_url:str, image file url or local image file path
    :return: llm response
    """
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": query,
            },
            {
                "type": "image_url",
                "image_url": image_url,
            }
        ]
    )
    return multimodal_llm.invoke(
        [message]
    )

#
# @tool
# def detect_obstacles(frame:cv2.typing.MatLike, net:cv2.dnn.Net, confidence_threshold:float) -> Tuple[cv2.typing.MatLike, List[Any]]:
#     """
#     :param frame:
#     :param net:
#     :param confidence_threshold:
#     :return:
#     """
#     (h, w) = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
#     net.setInput(blob)
#     detections = net.forward()
#     obstacles = []
#
#     for i in np.arange(0, detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > confidence_threshold:
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")
#             obstacles.append((startX, startY, endX, endY))
#             cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
#
#     return frame, obstacles
#
#
# @tool
# def avoidance_trajectory(frame, obstacles):
#     (h, w) = frame.shape[:2]
#     for (startX, startY, endX, endY) in obstacles:
#         centerX = (startX + endX) // 2
#         centerY = (startY + endY) // 2
#
#         if centerX < w // 3:
#             direction = "Move Right"
#             cv2.arrowedLine(frame, (centerX, centerY), (centerX + 50, centerY), (0, 0, 255), 2)
#         elif centerX > 2 * w // 3:
#             direction = "Move Left"
#             cv2.arrowedLine(frame, (centerX, centerY), (centerX - 50, centerY), (0, 0, 255), 2)
#         else:
#             direction = "Move Forward"
#             cv2.arrowedLine(frame, (centerX, centerY), (centerX, centerY - 50), (0, 0, 255), 2)
#
#         cv2.putText(frame, direction, (centerX - 50, centerY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#
#     return frame
#
#
# @tool
# def refine_threshold(obstacles, confidence_threshold, learning_rate=0.01):
#     if len(obstacles) == 0:
#         # No obstacles detected, decrease threshold slightly to be more sensitive
#         confidence_threshold = max(0.1, confidence_threshold - learning_rate)
#     else:
#         # Obstacles detected, increase threshold slightly to be more selective
#         confidence_threshold = min(0.9, confidence_threshold + learning_rate)
#     return confidence_threshold
#
#
# @tool
# def save_data(frame:cv2.typing.MatLike, obstacles, confidence_threshold):
#     timestamp = int(time.time())
#     image_path = f'data/image_{timestamp}.jpg'
#     cv2.imwrite(image_path, frame)
#
#     # Convert obstacles to a serializable format
#     obstacles_serializable = [(int(x1), int(y1), int(x2), int(y2)) for (x1, y1, x2, y2) in obstacles]
#
#     data = {
#         'timestamp': timestamp,
#         'confidence_threshold': float(confidence_threshold),
#         'obstacles': obstacles_serializable
#     }
#
#     with open(f'data/params_{timestamp}.json', 'w') as f:
#         json.dump(data, f)
#
#     return image_path, f'data/params_{timestamp}.json'
#
#
# @tool
# def collision_avoidance():
#     global confidence_threshold
#
#     # Create a data directory if it doesn't exist
#     os.makedirs('data', exist_ok=True)
#
#     # Open the video capture device (adjust the index if needed)
#     cap = cv2.VideoCapture(0)  # 0 is typically the default webcam; adjust if necessary
#
#     if not cap.isOpened():
#         print("Error: Couldn't open video stream.")
#         return
#
#     last_update_time = time.time()
#     frame_skip = 2  # Process every 2nd frame
#     frame_count = 0
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame")
#             break
#
#         frame_count += 1
#         if frame_count % frame_skip != 0:
#             continue
#
#         # Detect obstacles
#         frame_with_obstacles, obstacles = detect_obstacles(frame, net, confidence_threshold)
#
#         # Refine the confidence threshold based on recent detections
#         current_time = time.time()
#         if current_time - last_update_time > 1:  # Update every second
#             last_update_time = current_time
#             confidence_threshold = refine_threshold(obstacles, confidence_threshold)
#             print(f"Refining model... New confidence threshold: {confidence_threshold:.2f}")
#
#         # Determine avoidance trajectory
#         frame_with_trajectory = avoidance_trajectory(frame_with_obstacles, obstacles)
#
#         # Save the frame and parameters
#         save_data(frame_with_trajectory, obstacles, confidence_threshold)
#
#         # Display the resulting frame
#         cv2.imshow('Drone Live Stream', frame_with_trajectory)
#
#         # Press 'q' to exit the video stream
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     # Release the capture and close OpenCV windows
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# @tool
# def arm_vehicle():
#     """
#     Arms the vehicle and waits until the vehicle is armed.
#     """
#     while not vehicle.is_armable:
#         print("Waiting for vehicle to become armable...")
#         time.sleep(1)
#
#     print("Arming motors")
#     vehicle.mode = VehicleMode("GUIDED")
#     vehicle.armed = True
#
#     while not vehicle.armed:
#         print("Waiting for arming...")
#         time.sleep(1)
#     return "Vehicle is armed."
#
# @tool
# def takeoff(altitude:int):
#     """
#     Initiates takeoff to a specified altitude in int
#     :param altitude: altitude in int
#     :return: vehicle's final altitude
#     """
#     print("Taking off!")
#     vehicle.simple_takeoff(altitude)
#
#     while True:
#         print("Altitude: ", vehicle.location.global_relative_frame.alt)
#         if vehicle.location.global_relative_frame.alt >= altitude * 0.95:
#             print("Reached target altitude")
#             break
#         time.sleep(1)
#     return vehicle.location.global_relative_frame.alt
#
# @tool
# def land():
#     """
#     Lands the vehicle.
#     returns true or false for vehicle on state
#     """
#     print("Landing")
#     vehicle.mode = VehicleMode("LAND")
#     while vehicle.armed:
#         print("Waiting for landing...")
#         time.sleep(1)
#     return vehicle.armed
# @tool
# def move_ned_velocity(velocity_x:float, velocity_y:float, velocity_z:float, duration:int):
#     """
#     Move vehicle in direction based on specified velocity vectors given the following -
#     :param velocity_x: float, velocity in x direction
#     :param velocity_y: float, velocity in y direction
#     :param velocity_z: float, velocity in z direction
#     :param duration: int seconds
#     :return: true if operation successful else false
#     """
#
#     msg = vehicle.message_factory.set_position_target_local_ned_encode(
#         0,
#         0, 0,
#         mavutil.mavlink.MAV_FRAME_LOCAL_NED,
#         0b0000111111000111,  # bitmask to indicate which dimensions should be ignored
#         0, 0, 0,
#         velocity_x, velocity_y, velocity_z,
#         0, 0, 0,
#         0, 0)
#
#     for _ in range(0, duration):
#         vehicle.send_mavlink(msg)
#         time.sleep(1)
#
#
# @tool
# def get_location_metres(original_location, dNorth, dEast):
#     """
#     Returns a LocationGlobal object containing the latitude/longitude `dNorth` and `dEast` metres from the
#     specified `original_location`. The returned LocationGlobal has the same `alt` value
#     as `original_location`.
#
#     :param original_location:
#     :param dNorth:
#     :param dEast:
#     :return:
#     """
#     earth_radius = 6378137.0  # Radius of "spherical" earth
#     # Coordinate offsets in radians
#     dLat = dNorth / earth_radius
#     dLon = dEast / (earth_radius * math.cos(math.pi * original_location.lat / 180))
#
#     # New position in decimal degrees
#     newlat = original_location.lat + (dLat * 180 / math.pi)
#     newlon = original_location.lon + (dLon * 180 / math.pi)
#     if type(original_location) is LocationGlobal:
#         targetlocation = LocationGlobal(newlat, newlon, original_location.alt)
#     elif type(original_location) is LocationGlobalRelative:
#         targetlocation = LocationGlobalRelative(newlat, newlon, original_location.alt)
#     else:
#         raise Exception("Invalid Location object passed")
#
#     return targetlocation
#
# @tool
# def get_distance_metres(aLocation1, aLocation2):
#     """
#     Returns the ground distance in metres between two LocationGlobal objects.
#     """
#     dlat = aLocation2.lat - aLocation1.lat
#     dlong = aLocation2.lon - aLocation1.lon
#     return math.sqrt((dlat * dlat) + (dlong * dlong)) * 1.113195e5
#
# @tool
# def goto(dNorth, dEast, gotoFunction=vehicle.simple_goto):
#     """
#     Moves the vehicle to a position dNorth metres North and dEast metres East of the current position.
#     """
#     currentLocation = vehicle.location.global_relative_frame
#     targetLocation = get_location_metres(currentLocation, dNorth, dEast)
#     targetDistance = get_distance_metres(currentLocation, targetLocation)
#     gotoFunction(targetLocation)
#
# @tool
# def change_altitude(target_altitude):
#     """
#
#     :param target_altitude:
#     :return:
#     """
#     print("Changing altitude to:", target_altitude)
#     current_location = vehicle.location.global_relative_frame
#     target_location = LocationGlobalRelative(current_location.lat, current_location.lon, target_altitude)
#     vehicle.simple_goto(target_location)
#
#     while True:
#         print("Altitude: ", vehicle.location.global_relative_frame.alt)
#         if abs(vehicle.location.global_relative_frame.alt - target_altitude) < 0.5:
#             print("Reached target altitude")
#             break
#         time.sleep(1)


# create pilot agent
chat_history = []
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_PROMPT
        ),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# create a tool list

tools = [
    image_query
]

# bind llm with tools

llm_with_tools = multimodal_llm.bind_tools(tools)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

def transform(frame: av.VideoFrame):
    img = frame.to_ndarray(format="bgr24")


    return av.VideoFrame.from_ndarray(img, format="bgr24")


if __name__ == '__main__':
    webrtc_streamer(
        key="streamer",
        video_frame_callback=transform,
        sendback_audio=False
    )
