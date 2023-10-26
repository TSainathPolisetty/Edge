 //Working Code base level.
#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"


// Constants from training data stats
const float x_mean = 0.000118;
const float y_mean = -0.000537;
const float z_mean = 0.000103;

const float x_range = 0.719390 + 0.744416; // max - min
const float y_range = 0.603622 + 0.713384;
const float z_range = 0.903643 + 0.774497;

namespace {
  const tflite::Model* model = tflite::GetModel(dense_model_tflite);
  tflite::MicroInterpreter* interpreter;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  constexpr int kTensorArenaSize = 2000;
  alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}

// Normalize function
float normalize(float value, float mean, float range) {
    return (value - mean) / range;
}



void setupTFLite() {
    tflite::InitializeTarget();

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.print("Model provided is schema version ");
        Serial.print(model->version());
        Serial.print(" not equal to supported version ");
        Serial.println(TFLITE_SCHEMA_VERSION);
        return;
    }

    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, nullptr);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("AllocateTensors() failed");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);
}

void setup() {
    Serial.begin(115200);
    while (!Serial);
    delay(2000);
    Serial.println("Initializing...");

    if (!IMU.begin()) {
        Serial.println("Failed to initialize IMU!");
        return;
    }

    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, LOW);
    Serial.println("LED setup complete.");

    setupTFLite();
    Serial.println("TFLite setup complete.");
}

void loop() {
    if (Serial.available()) {
        char command = Serial.read();
        float x,y,z;

        switch (command) {
            case '1':  // Prediction using accelerometer
                if (IMU.accelerationAvailable()) {
                    IMU.readAcceleration(x, y, z);
                }
                break;
            case '2':  // Prediction using gyroscope
                if (IMU.gyroscopeAvailable()) {
                    IMU.readGyroscope(x, y, z);
                }
                break;
            case '3':  // Prediction using magnetometer
                if (IMU.magneticFieldAvailable()) {
                    IMU.readMagneticField(x, y, z);
                }
                break;
            default:
                Serial.println("Invalid Command!");
                return;
        }

        x = normalize(x, x_mean, x_range);
        y = normalize(y, y_mean, y_range);
        z = normalize(z, z_mean, z_range);

        input->data.f[0] = x;
        input->data.f[1] = y;
        input->data.f[2] = z;

        if (interpreter->Invoke() != kTfLiteOk) {
            Serial.println("Invoke failed");
            return;
        }

       
          float max_score = output->data.f[0];
          int predicted_label = 0;

          for (int i = 1; i < 4; i++) {
              if (output->data.f[i] > max_score) {
                  max_score = output->data.f[i];
                  predicted_label = i;
              }
          }

          if (predicted_label != 0) {
              for (int i = 0; i < predicted_label; i++) {
                  digitalWrite(LED_BUILTIN, HIGH);
                  delay(250);
                  digitalWrite(LED_BUILTIN, LOW);
                  delay(250);
              }
          }

        //Serial.print("Prediction result: ");
        //Serial.println(predicted_label);
        //delay(1000);
        //Serial.println(predicted_label);  // Send prediction result back to the laptop

        // Send the prediction result back to the base-station
        Serial.println(predicted_label);
    }
}








