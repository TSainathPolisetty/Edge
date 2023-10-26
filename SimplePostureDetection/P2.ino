#include <Arduino_LSM9DS1.h>

const int LED_PIN = 13;  // Onboard LED pin
float x, y, z;
int degreesX = 0;
int degreesY = 0;

void setup() {
  pinMode(LED_PIN, OUTPUT);
  Serial.begin(9600);
  while (!Serial);
  Serial.println("Started");

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  Serial.print("Accelerometer sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println("Hz");
}

void loop() {
  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(x, y, z);
  }

  Serial.print("Accelerometer sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println("Hz");
  Serial.print("x = "); Serial.println(x);
  Serial.print("y = "); Serial.println(y);
  Serial.print("z = "); Serial.println(z);

  if (isSupine(x, y, z)) {
    blinkLED(1);
  } else if (isProne(x, y, z)) {
    blinkLED(2);
  } else if (isSide(x, y, z)) {
    blinkLED(3);
  } else {
    digitalWrite(LED_PIN, LOW);
  }

  delay(1000);
}

bool isSupine(float x, float y, float z) {
  return z > 0.6 && z <= 1.3;  // Restricting the range for supine
}

bool isProne(float x, float y, float z) {
  return z < -0.6 && z >= -1.3;  // Restricting the range for prone
}

bool isSide(float x, float y, float z) {
  return (y > 0.6 && y <= 1.3) || (y < -0.6 && y >= -1.3);  // Restricting the range for side
}

void blinkLED(int times) {
  for (int i = 0; i < times; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(300);
    digitalWrite(LED_PIN, LOW);
    delay(300);
  }
  delay(800);
}
