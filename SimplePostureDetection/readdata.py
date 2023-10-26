import serial
import csv
import time

# Setup the serial connection
PORT = '/dev/ttyACM0'  # Arduino's port
BAUD_RATE = 115200 # Baud rate same as mentioned in the arduino code
ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
ser.flush()

#Data collection code
postures = ['side', 'supine', 'prone', 'sitting']
current_posture = ''

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
            
            try:
                while True:
                    line = ser.readline().decode('utf-8').strip()
                    if line:
                        x, y, z = map(float, line.split(','))
                        writer.writerow([x, y, z])
                    time.sleep(0.1)  # To match Arduino's delay
            except KeyboardInterrupt:
                print(f"Data collection for {current_posture} posture stopped.")
    else:
        print("Invalid choice!")

ser.close()
