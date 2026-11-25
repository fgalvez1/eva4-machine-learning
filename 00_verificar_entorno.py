import sys, tensorflow as tf, numpy, sklearn, matplotlib
print("Python:", sys.version)
print("TensorFlow:", tf.__version__)
print("GPU disponible:", tf.config.list_physical_devices("GPU") or "No (se entrenará con CPU)")
print("NumPy:", numpy.__version__, "| scikit-learn:", sklearn.__version__, "| Matplotlib:", matplotlib.__version__)