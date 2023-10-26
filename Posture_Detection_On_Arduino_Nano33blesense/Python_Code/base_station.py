import serial
import time

# Mapping of predicted labels to classes
LABEL_MAP = {
    "0": "Unknown",
    "1": "Supine",
    "2": "Side",
    "3": "Prone",
    "4": "Side"
}

def main():
    # Attempt to establish a connection
    try:
        ser = serial.Serial('/dev/ttyACM0', 115200)
    except serial.SerialException:
        print("Error: Unable to establish a connection with the device. Please ensure it's connected and try again.")
        return

    print("\n==============================")
    print("  Real-time Prediction Tool  ")
    print("==============================\n")
    print("Setting up connection with the device...")

    # Wait for the Arduino to initialize
    time.sleep(2)

    # Clear any existing buffered data
    ser.reset_input_buffer()

    while True:
        print("\nSelect the sensor for prediction:")
        print("1: Accelerometer")
        print("2: Gyroscope")
        print("3: Magnetometer")
        print("q: Quit")

        choice = input("\nEnter your choice: ")

        if choice == 'q':
            break

        if choice in ['1', '2', '3']:
            # Send the sensor choice to Arduino
            ser.write(choice.encode('utf-8'))
            print("\nCollecting data and running inference...")

            # Read the prediction result from Arduino
            full_result = ser.readline().decode('utf-8').strip()

            # Map the label to the corresponding class and print
            class_name = LABEL_MAP.get(full_result, "Unknown")  # Default to "Unknown" if label is not in the map
            print(f"\nResult: Your current posture is '{class_name}'.\n")

        else:
            print("\nInvalid choice. Please select a valid option.")

    # Close the serial connection
    ser.close()

    print("\nThank you for using the Real-time Prediction Tool!")
    print("==============================\n")

if __name__ == "__main__":
    main()


