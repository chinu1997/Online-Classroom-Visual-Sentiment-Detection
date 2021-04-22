<h1>Introduction</h1>

This project aims to classify the emotion on a person's face into one of seven categories, using deep convolutional neural networks. The model is trained on the FER-2013 dataset which was published on International Conference on Machine Learning (ICML). This dataset consists of 35887 grayscale, 48x48 sized face images with seven emotions - angry, disgusted, fearful, happy, neutral, sad and surprised.

<h1>Dataset</h1>

Fro this project we have kaggle dataset fer 2013: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data.

dataset_shape=(35887,3)

After reshaping for data:-

train shape (28709, 48, 48, 1)
validation shape (3589, 48, 48, 1)
validation shape (3589, 48, 48, 1)


<h1>Experiment With Model</h1>
  <h3>1.Using MLP with tensorflow version 2.4.1</h3>
Epoch 1/5
898/898 [==============================] - 4s 4ms/step - loss: 47.1633 - accuracy: 0.2087 - val_loss: 2.3757 - val_accuracy: 0.2282

Epoch 2/5
898/898 [==============================] - 3s 4ms/step - loss: 1.9894 - accuracy: 0.2493 - val_loss: 2.1619 - val_accuracy: 0.1828

Epoch 3/5
898/898 [==============================] - 3s 4ms/step - loss: 1.8900 - accuracy: 0.2655 - val_loss: 2.6126 - val_accuracy: 0.1167

Epoch 4/5
898/898 [==============================] - 3s 4ms/step - loss: 1.8567 - accuracy: 0.2327 - val_loss: 1.8168 - val_accuracy: 0.2449

Epoch 5/5
898/898 [==============================] - 3s 4ms/step - loss: 1.8110 - accuracy: 0.2486 - val_loss: 1.8177 - val_accuracy: 0.2449
<tensorflow.python.keras.callbacks.History at 0x7fd02058f390>

<h3>2.Using CNN</h3> 

Epoch 1/15
684/684 [==============================] - 56s 82ms/step - loss: 0.6443 - accuracy: 0.7727 - val_loss: 3.1014 - val_accuracy: 0.4260

Epoch 2/15
684/684 [==============================] - 56s 81ms/step - loss: 0.5620 - accuracy: 0.8021 - val_loss: 3.4424 - val_accuracy: 0.3954

Epoch 3/15
684/684 [==============================] - 56s 81ms/step - loss: 0.5157 - accuracy: 0.8240 - val_loss: 3.6780 - val_accuracy: 0.4124

Epoch 4/15
684/684 [==============================] - 56s 82ms/step - loss: 0.4715 - accuracy: 0.8365 - val_loss: 4.1690 - val_accuracy: 0.4154

Epoch 5/15
684/684 [==============================] - 56s 82ms/step - loss: 0.4663 - accuracy: 0.8418 - val_loss: 4.1743 - val_accuracy: 0.4118

Epoch 6/15
684/684 [==============================] - 56s 82ms/step - loss: 0.4277 - accuracy: 0.8535 - val_loss: 4.6127 - val_accuracy: 0.3996

Epoch 7/15
684/684 [==============================] - 56s 82ms/step - loss: 0.3749 - accuracy: 0.8723 - val_loss: 4.8611 - val_accuracy: 0.4093

Epoch 8/15
684/684 [==============================] - 56s 82ms/step - loss: 0.3612 - accuracy: 0.8795 - val_loss: 5.1365 - val_accuracy: 0.3929

Epoch 9/15
684/684 [==============================] - 56s 82ms/step - loss: 0.3314 - accuracy: 0.8880 - val_loss: 5.3357 - val_accuracy: 0.4018

Epoch 10/15
684/684 [==============================] - 56s 81ms/step - loss: 0.3101 - accuracy: 0.8975 - val_loss: 5.6454 - val_accuracy: 0.4163

Epoch 11/15
684/684 [==============================] - 56s 82ms/step - loss: 0.2928 - accuracy: 0.9048 - val_loss: 5.8554 - val_accuracy: 0.4082

Epoch 12/15
684/684 [==============================] - 56s 82ms/step - loss: 0.3159 - accuracy: 0.8982 - val_loss: 6.0766 - val_accuracy: 0.4104

Epoch 13/15
684/684 [==============================] - 56s 82ms/step - loss: 0.2567 - accuracy: 0.9161 - val_loss: 6.5591 - val_accuracy: 0.4132

Epoch 14/15
684/684 [==============================] - 56s 82ms/step - loss: 0.2475 - accuracy: 0.9221 - val_loss: 6.6915 - val_accuracy: 0.4082

Epoch 15/15
684/684 [==============================] - 56s 82ms/step - loss: 0.2542 - accuracy: 0.9191 - val_loss: 7.1066 - val_accuracy: 0.4065

<tensorflow.python.keras.callbacks.History at 0x7fd01a0b55
  


