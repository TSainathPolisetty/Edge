# import tensorflow as tf

# # Load the trained model
# model = tf.keras.models.load_model("lstm_model_relu.h5")

# # Convert to TFLite
# converter = tf.lite.TFLiteConverter.from_keras_model(model)

# # Enable default optimizations, which include quantization
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# # Specify full integer quantization
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.int8
# converter.inference_output_type = tf.int8

# # Provide a representative dataset
# def representative_dataset():
#     # Load your data
#     # Here, I'm making an assumption based on your previous code. Adjust accordingly.
#     train_data = pd.read_csv("train_data.csv").values
#     X_train, y_train = train_data[:, :-1], train_data[:, -1]
#     X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

#     for i in range(len(X_train)):
#         yield [X_train[i:i+1]]

# converter.representative_dataset = representative_dataset

# # Convert and save the quantized model
# tflite_quantized_model = converter.convert()
# # Save the TFLite model

# with open('lstm_model_quantized.tflite', 'wb') as f:
#     f.write(tflite_quantized_model)


import tensorflow as tf

def convert_to_tflite(model_name):
    # Load the model
    model = tf.keras.models.load_model(model_name)
    
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Set the flags as suggested by the error message
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, 
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False
    
    tflite_model = converter.convert()
    
    # Save the TFLite model
    with open(f"{model_name.split('.')[0]}.tflite", 'wb') as f:
        f.write(tflite_model)
        
    print(f"{model_name.split('.')[0]}.tflite saved successfully!")

# Since you're no longer iterating over activation functions
# you can directly convert your simple feed-forward model
model_name = "dense_model.h5"
convert_to_tflite(model_name)
