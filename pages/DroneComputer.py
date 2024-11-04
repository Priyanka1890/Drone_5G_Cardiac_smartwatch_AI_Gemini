import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes
import numpy as np
import av
import folium
import streamlit as st
from streamlit_folium import st_folium
from dronekit import connect, VehicleMode, LocationGlobalRelative, LocationGlobal
import time
from pymavlink import mavutil
import math


connection_string = '/dev/ttyAMA0'
baud_rate = 57600
# Connect to the vehicle
try:
    v = None
    # Connect to the vehicle
    # vehicle = connect('127.0.0.1:14550', wait_ready=True)
    vehicle = connect(connection_string, baud=baud_rate, wait_ready=True)
except:
    vehicle = None


def transform(frame: av.VideoFrame):
    # cv2.imwrite("./test.jpeg", frame)
    img = frame.to_ndarray(format="bgr24")

    if filter == "blur":
        img = cv2.GaussianBlur(img, (21, 21), 0)
    elif filter == "canny":
        img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
    elif filter == "grayscale":
        # We convert the image twice because the first conversion returns a 2D array.
        # the second conversion turns it back to a 3D array.
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    elif filter == "sepia":
        kernel = np.array(
            [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]
        )
        img = cv2.transform(img, kernel)
    elif filter == "invert":
        img = cv2.bitwise_not(img)
    elif filter == "capture":
        print("asdbujhabsdhbajshdbhbasdhbahbshdb")
        cv2.imwrite(f"./received_frame.jpg", frame)
    elif filter == "none":
        pass

    return av.VideoFrame.from_ndarray(img, format="bgr24")



def arm_vehicle():
    """
    Arms the vehicle and waits until the vehicle is armed.
    """
    if vehicle:
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
    else:
        return "connect to vehicle"

def takeoff(altitude):
    """
    Initiates takeoff to a specified altitude.
    """
    if vehicle:
        print("Taking off!")
        vehicle.simple_takeoff(altitude)

        while True:
            print("Altitude: ", vehicle.location.global_relative_frame.alt)
            if vehicle.location.global_relative_frame.alt >= altitude * 0.95:
                print("Reached target altitude")
                break
            time.sleep(1)
    else:
        return "connect to vehicle"

def land():
    """
    Lands the vehicle.
    """
    if vehicle:
        print("Landing")
        vehicle.mode = VehicleMode("LAND")
        while vehicle.armed:
            print("Waiting for landing...")
            time.sleep(1)

    else:
        return "connect to vehicle"

def move_ned_velocity(velocity_x, velocity_y, velocity_z, duration):
    """
    Move vehicle in direction based on specified velocity vectors.
    """
    if vehicle:
        msg = vehicle.message_factory.set_position_target_local_ned_encode(
            0,
            0, 0,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111000111,  # bitmask to indicate which dimensions should be ignored
            0, 0, 0,
            velocity_x, velocity_y, velocity_z,
            0, 0, 0,
            0, 0)

        for _ in range(0, duration):
            vehicle.send_mavlink(msg)
            time.sleep(1)
    else:
        return "connect to vehicle"

def get_location_metres(original_location, dNorth, dEast):
    """
    Returns a LocationGlobal object containing the latitude/longitude `dNorth` and `dEast` metres from the
    specified `original_location`. The returned LocationGlobal has the same `alt` value
    as `original_location`.
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

def get_distance_metres(aLocation1, aLocation2):
    """
    Returns the ground distance in metres between two LocationGlobal objects.
    """
    dlat = aLocation2.lat - aLocation1.lat
    dlong = aLocation2.lon - aLocation1.lon
    return math.sqrt((dlat * dlat) + (dlong * dlong)) * 1.113195e5

def goto(dNorth, dEast, gotoFunction=vehicle.simple_goto):
    """
    Moves the vehicle to a position dNorth metres North and dEast metres East of the current position.
    """
    if vehicle:
        currentLocation = vehicle.location.global_relative_frame
        targetLocation = get_location_metres(currentLocation, dNorth, dEast)
        targetDistance = get_distance_metres(currentLocation, targetLocation)
        gotoFunction(targetLocation)
    else:
        return "connect to vehicle"

def change_altitude(target_altitude):
    """
    Changes the altitude of the vehicle to the specified target altitude.
    """
    if vehicle:
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
    else:
        return "connect to vehicle"

def main():


    st.title("DroneComputer")
    usr_data = None
    if 'user_data' in st.session_state and st.session_state['user_data'] != "INVALID":
        usr_data = st.session_state['user_data']
    else:
        st.switch_page("home.py")

    tab1, tab2, tab3 = st.tabs([f"{usr_data['username']}'s Data", "Drone Simulator", "Logout"])
    
    

    with tab1:

        

        user_data = ["Pdy", "priyanka", "pdey1", "pd", "pdey"]
        alarm_data =["Safe", "Safe", "Safe", "Safe"]
        lat_data = ["NA", "NA", "NA", "NA"]
        lon_data = ["NA", "NA", "NA","NA"]

        df = pd.read_csv("user_location.csv")
        lat = df.lat.tolist()[0]
        lat_data.append(lat)
        lon = df.lon.tolist()[0]
        lon_data.append(lon)
        df = pd.read_csv("hrate.csv")
        hrate = df.value.tolist()[-1]
        min_al = df.min_alarm.tolist()[-1]
        max_al = df.max_alarm.tolist()[-1]


        if (hrate <= max_al) & (hrate >= min_al):
            alarm_data.append("Safe")
        else:
            alarm_data.append("Danger")

        df = pd.DataFrame({
            "idx":[1, 2, 3, 4, 5],
            "user_name":user_data,
            "user_loc_lat":lat_data,
            "user_loc_lon": lon_data,
            "alarm":alarm_data
        })
        st.markdown("### User List ")
        st.dataframe(df)


    with tab2:

        c1, c2 = st.columns(2)

        with c1:
            m = folium.Map(location=[39.949610, -75.150282], zoom_start=16)
            folium.Marker(
                [39.949610, -75.150282], popup="Drone", tooltip="Drone"
            ).add_to(m)
            folium.Marker(
                [39.849610, -75.150282], popup="User", tooltip="User"
            ).add_to(m)
            # call to render Folium map in Streamlit
            st_data = st_folium(m, width=725)
        with c2:
            st.markdown(":blue[Drone Camera]")
            webrtc_streamer(
                key="streamer",
                video_frame_callback=transform,
                sendback_audio=True
            )

        with st.container(border=True):
            arm_drone = st.button("ARM Drone", use_container_width=True)
            if arm_drone:
                status = arm_vehicle()
                st.write(status)
            takeoff_alt = st.slider("Takeoff Alt.", 10, 100, 5)
            if takeoff_alt:
                takeoff(takeoff_alt)
            land = st.button("Land", use_container_width=True)
            if land:
                land()
            go_north = st.slider("Goto North", 10, 100, 5)
            go_east = st.slider("Goto East", 10, 100, 5)
            goto_dicr = st.button("Goto", use_container_width=True)
            if goto_dicr:
                goto(dNorth=go_north, dEast=go_east)



        #markdown = """
        #The drone computer is connected with drone via Mission Planner (https://ardupilot.org/planner/)
        #"""
        #
        #    st.markdown(markdown)
        #    components.html(
        #    '''
        #    <iframe
        #    src="https://vnc.eu1.pitunnel.com/novnc/vnc.html?autoconnect=1&resize=scale&quality=5&compression=7&show_dot=1&path=PGuJOQwq5Fx9lsYPC97PG7qnWLm3oMCD2VJsdHfHeP7qmIMsK94e4Tm2UpfdGBq7RIGp8Nb5NCPc2x7Tr5aexakS9hYPSMzvrlNBanYlWATDC592ZvhYQThKKjPhpEgR/websockify?embed=true"
        #        height=500 width=750></iframe>
        #     ''',
        #    height=900,
        #     width=900
       #)


    with tab3:
        if st.button("Logout"):
            st.session_state['user_data'] = "INVALID"
            st.switch_page("home.py")








if __name__ == '__main__':
    main()

