from dronekit import connect, VehicleMode, LocationGlobalRelative, LocationGlobal
import time
from pymavlink import mavutil
import math


connection_string = '/dev/ttyAMA0'
baud_rate = 57600


# Connect to the vehicle
vehicle = connect('127.0.0.1:14550', wait_ready=True)
#vehicle = connect(connection_string, baud=baud_rate, wait_ready=True)


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
    print("Vehicle is armed.")

def takeoff(altitude):
    """
    Initiates takeoff to a specified altitude.
    """
    print("Taking off!")
    vehicle.simple_takeoff(altitude)

    while True:
        print("Altitude: ", vehicle.location.global_relative_frame.alt)
        if vehicle.location.global_relative_frame.alt >= altitude * 0.95:
            print("Reached target altitude")
            break
        time.sleep(1)

def land():
    """
    Lands the vehicle.
    """
    print("Landing")
    vehicle.mode = VehicleMode("LAND")
    while vehicle.armed:
        print("Waiting for landing...")
        time.sleep(1)

def move_ned_velocity(velocity_x, velocity_y, velocity_z, duration):
    """
    Move vehicle in direction based on specified velocity vectors.
    """
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
    currentLocation = vehicle.location.global_relative_frame
    targetLocation = get_location_metres(currentLocation, dNorth, dEast)
    targetDistance = get_distance_metres(currentLocation, targetLocation)
    gotoFunction(targetLocation)

def change_altitude(target_altitude):
    """
    Changes the altitude of the vehicle to the specified target altitude.
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

# Example usage
arm_vehicle()  # Arm the vehicle
takeoff(10)  # Takeoff to 10 meters
time.sleep(10)
goto(-15, -5)
time.sleep(10)
goto(-30, -15)
time.sleep(10)
goto(-10, 0)
time.sleep(10)

# Change altitude to 20 meters
change_altitude(5)
time.sleep(10)

# Land
land()

# Close vehicle object before exiting script
vehicle.close()
print("Completed")

