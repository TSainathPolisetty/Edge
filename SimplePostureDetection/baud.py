import serial
import time

# Modify the port to match your Arduino's port.
port_name = "/dev/ttyACM0"
baud_rates = [9600, 14400, 19200, 38400, 57600, 115200, 230400, 250000, 500000, 1000000]

for baud in baud_rates:
    with serial.Serial(port_name, baudrate=baud, timeout=1) as ser:
        time.sleep(2)  # Allow time for connection.
        ser.write(b't')
        response = ser.read(2)
        if response == b'HL':
            print(f"Baud rate {baud} is OK!")
        else:
            print(f"Baud rate {baud} failed!")
            break
