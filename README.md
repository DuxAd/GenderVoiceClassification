# Audio Gender Recognition with CNN

This project implements a Deep Learning model capable of classifying the gender of a speaker (Male/Female) from `.wav` audio files. It achieves **~99% accuracy** on the test set and has been validated on "in-the-wild" audio samples from diverse sources like YouTube. The dataset used for the training can be find here : https://www.kaggle.com/datasets/murtadhanajim/gender-recognition-by-voiceoriginal/data

## Features
* **Preprocessing**: Real-time Mel-spectrogram computation using TensorFlow's signal processing tools.
* **Advanced Data Augmentation**: Includes circular shifts (`tf.roll`), random noise, and gain adjustment to ensure model generalization.
* **Custom CNN Architecture**: Uses a dual-path input layer to capture both temporal and spectral features.
* **Inference Pipeline**: A dedicated prediction script that handles long audio files by segmenting them into 3-second windows.

---

## Architecture
The model is built with Keras and follows a specific design for audio:
* **Time-Frequency Feature Extraction**: The first layer consists of two parallel `Conv2D` branches:
    * **Time branch**: Kernel (1, 5) to capture rhythmic patterns.
    * **Frequency branch**: Kernel (5, 1) to capture tonal/pitch characteristics.
* **Feature Fusion**: The branches are concatenated, followed by `BatchNormalization` and `MaxPooling2D`.
* **Classification Head**: Flattened features passed through a `Dense` layer (32 units) with `Dropout (0.3)` to prevent overfitting.

---

## Results
The model shows high stability and excellent generalization:
* **Accuracy**: ~99% on the test dataset.
* **Loss**: Optimized using `RMSprop` with `SparseCategoricalCrossentropy`.
* **Validation**: Confirmed with a confusion matrix showing near-perfect separation between classes.

<img width="306" height="234" alt="image" src="https://github.com/user-attachments/assets/8919adc0-c76e-4ef4-a298-626670d72360" />
<img width="306" height="234" alt="image" src="https://github.com/user-attachments/assets/4440617a-7819-4773-9a37-e4855dfc75e9" />
<img width="306" height="234" alt="image" src="https://github.com/user-attachments/assets/a02910db-676e-4964-876a-5e57a0f87e31" />
<img width="306" height="234" alt="image" src="https://github.com/user-attachments/assets/396d095d-e4da-44f8-ae24-6f0cd25bfd73" />


---

