# import tensorflow as tf
#
#
# model =  tf.keras.models.load_model('/home/luca/raspicar/training/02-05-2024_14-44/checkpoints/cp-0025.keras')
# tf.saved_model.save(model, "/home/luca/raspicar/training/02-05-2024_14-44/checkpoints/saved_model")
#
#
# # Now convert the last/best model to tf lite
# #converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter = tf.lite.TFLiteConverter.from_saved_model('/home/luca/raspicar/training/02-05-2024_14-44/checkpoints/saved_model')
# #converter.optimizations = [tf.lite.Optimize.DEFAULT]
# #converter.target_spec.supported_types = [tf.float16]  # TODO maybe change optimization
# tflite_model = converter.convert()
# # Save the model.
# with open('/home/luca/raspicar/training/02-05-2024_14-44/checkpoints/model.tflite', 'wb') as f:
#     f.write(tflite_model)



import tensorflow as tf


# Convert the model.
model = tf.keras.models.load_model('/home/luca/raspicar/training/04-05-2024_00-24/checkpoints/cp-0010.keras')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('/home/luca/raspicar/training/04-05-2024_00-24/checkpoints/cp-0010.tflite', 'wb') as f:
  f.write(tflite_model)