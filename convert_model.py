import tensorflow as tf

model = tf.keras.models.load_model("sign_model.h5")


model.save("sign_model.keras")


