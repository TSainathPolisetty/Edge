import serial
import csv
import time

# Setup the serial connection
PORT = '/dev/ttyACM0'  # Arduino's port
BAUD_RATE = 115200 # Baud rate same as mentioned in the Arduino code
ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
ser.flush()

# Data collection code
postures = ['side', 'supine', 'prone', 'sitting']
current_posture = ''

def collect_data():
    # Collect data for each sensor type, return as a dictionary
    data = {}
    for _ in range(3):  # Since we have three sensors
        line = ser.readline().decode('utf-8').strip()
        sensor_type, values = line.split(': ')
        
        if sensor_type == "ACCEL":
            data[sensor_type] = list(map(float, values.split(',')))
        else:
            data[sensor_type] = list(map(float, values.split('\t')))
    return data

while True:
    print("Choose a posture for data collection:")
    for idx, posture in enumerate(postures):
        print(f"{idx}. {posture}")
    choice = input("Enter the number (or 'q' to quit): ")

    if choice == 'q':
        break
    elif choice in ['0', '1', '2', '3']:
        current_posture = postures[int(choice)]
        print(f"Collecting data for {current_posture} posture. Press 'CTRL+C' to stop...")
        
        with open(f"{current_posture}.csv", "a", newline='') as file:
            writer = csv.writer(file)
            # Write header if the file is empty
            if file.tell() == 0:
                writer.writerow(["Timestamp", "AccelX", "AccelY", "AccelZ", "GyroX", "GyroY", "GyroZ", "MagX", "MagY", "MagZ", "Posture"])

            try:
                while True:
                    data = collect_data()
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    writer.writerow([timestamp] + data["ACCEL"] + data["GYRO"] + data["MAG"] + [current_posture])
                    time.sleep(0.1)  # To match Arduino's delay
            except KeyboardInterrupt:
                print(f"Data collection for {current_posture} posture stopped.")
    else:
        print("Invalid choice!")

ser.close()
