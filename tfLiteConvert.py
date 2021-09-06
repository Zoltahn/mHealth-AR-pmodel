import tensorflow as tf

def convertModelDirectory(inModelDir, savedName = 'model'):
    converter = tf.lite.TFLiteConverter.from_saved_model("CNN_test_WISDM_ar/")
    tfLiteModel = converter.convert()
    
    with open(savedName, 'wb') as f:
         f.write(tfLiteModel)

def convertModelKeras(inModel, savedName = 'model'):
    converter = tf.lite.TFLiteConverter.from_keras_model(inModel)
    tfLiteModel = converter.convert()
    
    with open(savedName, 'wb') as f:
         f.write(tfLiteModel)
    