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
