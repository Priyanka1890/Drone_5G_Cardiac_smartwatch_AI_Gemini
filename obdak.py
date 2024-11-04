
import cv2
import numpy as np
import time
import os
import json

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




#
# # Define the paths to the model files
# proto_path = './MobileNetSSD_deploy.prototxt'
# model_path = './MobileNetSSD_deploy.caffemodel'
#
#
# # Load the MobileNetSSD model
# net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
#
# confidence_threshold = 0.5
#
# def detect_obstacles(frame, net, confidence_threshold):
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
# def avoidance_trajectory(frame, obstacles, decision):
#     (h, w) = frame.shape[:2]
#     for (startX, startY, endX, endY) in obstacles:
#         centerX = (startX + endX) // 2
#         centerY = (startY + endY) // 2
#
#         if centerX < w // 3:
#             direction = f"{decision}"
#             cv2.arrowedLine(frame, (centerX, centerY), (centerX + 50, centerY), (0, 0, 255), 2)
#         elif centerX > 2 * w // 3:
#             direction = f"{decision}"
#             cv2.arrowedLine(frame, (centerX, centerY), (centerX - 50, centerY), (0, 0, 255), 2)
#         else:
#             direction = f"{decision}"
#             cv2.arrowedLine(frame, (centerX, centerY), (centerX, centerY - 50), (0, 0, 255), 2)
#
#         cv2.putText(frame, direction, (centerX - 50, centerY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#
#     return frame
#
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
#
# def save_data(frame, obstacles, confidence_threshold):
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


def move_drone_by_control(coltrol_signals:list[str]):
    for e in coltrol_signals:
        if "stay" in e:
            print(e)
        elif "up" in e.strip():
            print("DRONE - up(5cm)")
        elif "down" in e.strip():
            print("DRONE - down(5cm)")
        elif "right" in e.strip():
            print("DRONE - right(5cm)")
        elif "straight" in e.strip():
            print("DRONE - straight(5)")
        elif "back" in e.strip():
            print("DRONE - back(5)")
        elif "left" in e.strip():
            print("DRONE - left(5cm)")

def main():
    global confidence_threshold

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
            response = multimodal_llm.invoke(
                [message]
            ).content
            print("maneuver ===>", response)
            controls = response.split(" ")

            move_drone_by_control(coltrol_signals=controls)
            import sys
            if time_window > 100:
                sys.exit(0)

            image_data = os.path.join(data_output_dir, f"cam_{timestamp}.txt")
            with open(image_data, "w") as f:
                f.write(response)
        time_window += 1
        if not ret:
            print("Failed to grab frame")
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Detect obstacles
        # frame_with_obstacles, obstacles = detect_obstacles(frame, net, confidence_threshold)
        #
        # # Refine the confidence threshold based on recent detections
        # current_time = time.time()
        # if current_time - last_update_time > 1:  # Update every second
        #     last_update_time = current_time
        #     confidence_threshold = refine_threshold(obstacles, confidence_threshold)
        #     print(f"Refining model... New confidence threshold: {confidence_threshold:.2f}")
        #
        # # Determine avoidance trajectory
        # frame_with_trajectory = avoidance_trajectory(frame_with_obstacles, obstacles, decision=response)
        #
        # # Save the frame and parameters
        # save_data(frame_with_trajectory, obstacles, confidence_threshold)
        #
        # # Display the resulting frame
        # cv2.imshow('Drone Live Stream', frame_with_trajectory)
        #
        # # Press 'q' to exit the video stream
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Release the capture and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
