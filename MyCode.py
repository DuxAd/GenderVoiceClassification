import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import tensorflow as tf
import sklearn
from keras.layers import Conv2D, Flatten, Dense, Input, BatchNormalization, Dropout, MaxPooling2D, Concatenate
from keras import Model
import pickle
import seaborn as sns
import librosa


path = "\GenderRecognition\data" # Path to the training and validation dataset #

ds_train = tf.keras.utils.audio_dataset_from_directory(
    directory=path+"_train",
    batch_size=None,
    output_sequence_length=48000, 
    seed=42                       
)

ds_val = tf.keras.utils.audio_dataset_from_directory(
    directory=path+"_test",
    batch_size=None,
    output_sequence_length=48000,
    seed=42
)

val_batches = tf.data.experimental.cardinality(ds_val).numpy()
ds_test = ds_val.take(val_batches//2)
ds_val = ds_val.skip(val_batches//2)

label_name = ds_train.class_names

def preprocess(audio, label):
    audio = tf.cast(audio, tf.float32)

    audio = audio / (tf.reduce_max(tf.abs(audio)) + 1e-6)
    audio_flat = tf.squeeze(audio, axis=-1)
    return audio_flat, label

def add_noise(audio, label):

    shift = tf.random.uniform([], -8000, 8000, dtype=tf.int32)
    audio = tf.roll(audio, shift=shift, axis=0)

    noise = tf.random.normal(tf.shape(audio), stddev=tf.random.uniform([], 0.001, 0.005))
    audio = audio + noise
    
    gain = tf.random.uniform([], 0.7, 1.3)
    audio = audio * gain
    
    audio = tf.clip_by_value(audio, -1.0, 1.0)

    return audio, label


def Spectrogram(audio, label):

    stft = tf.signal.stft(audio, frame_length=320, frame_step=160)
    stft_spectrogram = tf.abs(stft)
      
    mel_spectrogram = tf.matmul(stft_spectrogram, mel_matrix)

    spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
       
    return spectrogram, label

mel_matrix = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins = 64,
    num_spectrogram_bins = 257,
    sample_rate = 16000,
    lower_edge_hertz=20.0,
    upper_edge_hertz=8000.0
    )

ds_train_ready = ds_train.map(preprocess,num_parallel_calls=tf.data.AUTOTUNE) \
    .map(add_noise,num_parallel_calls=tf.data.AUTOTUNE) \
    .map(Spectrogram,num_parallel_calls=tf.data.AUTOTUNE) \
    .shuffle(buffer_size=1000) \
    .batch(128) \
    .prefetch(tf.data.AUTOTUNE)
    
ds_val_ready = ds_val.map(preprocess,num_parallel_calls=tf.data.AUTOTUNE) \
    .map(Spectrogram,num_parallel_calls=tf.data.AUTOTUNE) \
    .batch(128) \
    .prefetch(tf.data.AUTOTUNE)
     
ds_test_ready = ds_test.map(preprocess,num_parallel_calls=tf.data.AUTOTUNE) \
    .map(Spectrogram,num_parallel_calls=tf.data.AUTOTUNE) \
    .batch(128) \
    .prefetch(tf.data.AUTOTUNE)   
    
class AudioCNNModel:
   def __init__(self, num_classes):
       self.num_classes = num_classes

       self.training_accuracy = []
       self.training_loss = []
       self.validation_accuracy = []
       self.validation_loss = []

   def model(self): 
       inputs = Input(shape=(299, 64, 1), name='Inputs')

       x_time = Conv2D(filters=16, kernel_size=(1,5), strides = 1, padding='same', activation='relu')(inputs)
       x_frequency = Conv2D(filters=16, kernel_size=(5,1), strides = 1, padding='same', activation='relu')(inputs)
       x = Concatenate()([x_time, x_frequency])
       x = BatchNormalization()(x)
       x = MaxPooling2D(pool_size=(2, 2))(x)
       
       x = Conv2D(filters=32, kernel_size=3, strides = 1, padding='same', activation='relu')(x)
       x = BatchNormalization()(x)
       x = MaxPooling2D(pool_size=(2, 2))(x)
         
       # x = Conv2D(filters=64, kernel_size=3, strides = 1, padding='same', activation='relu')(x)
       # x = BatchNormalization()(x)
       # x = MaxPooling2D(pool_size=(2, 2))(x)
         
       # x = Conv2D(filters=128, kernel_size=3, strides = 1, padding='same', activation='relu')(x)
       # x = BatchNormalization()(x)
       # x = MaxPooling2D(pool_size=(2, 2))(x)
         
       x = Flatten()(x)
       x = Dense(32, activation='relu')(x)
         
       x = Dropout(0.3)(x)
       outputs = Dense(self.num_classes, activation='softmax')(x)
         
       self.model = Model(inputs, outputs, name="AudioCNN")
       
   def f_compile(self, learning_rate):

       optimizer=tf.keras.optimizers.RMSprop(learning_rate = learning_rate)
       self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

   def f_train(self, ds_train, ds_test):
       from keras.callbacks import EarlyStopping
       
       history = self.model.fit(
           x=ds_train,
           validation_data = ds_test,
            shuffle = True,
            epochs=15,
            callbacks=[EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)]
            )
       
       self.training_accuracy.extend(history.history['accuracy'])
       self.training_loss.extend(history.history['loss'])
       self.validation_accuracy.extend(history.history['val_accuracy'])
       self.validation_loss.extend(history.history['val_loss'])

############ Model ############
Audio_Model = AudioCNNModel(num_classes = 2)
print("\nParameters initialized")
Audio_Model.model()
print("\nModel created")
Audio_Model.f_compile(learning_rate = 0.001)
print("\nModel compile")

loading = 0
if loading:
    file = 'Model_Audio'
    with open(file +'.pkl', 'rb') as f:
        loaded_attributes = pickle.load(f)

    Audio_Model.training_accuracy = loaded_attributes['training_accuracy']
    Audio_Model.training_loss = loaded_attributes['training_loss']
    Audio_Model.validation_accuracy = loaded_attributes['validation_accuracy']
    Audio_Model.validation_loss = loaded_attributes['validation_loss']

    Audio_Model.model.load_weights(file + '.keras')
    print("\nLoaded")

print("\nTraining ... ")
Audio_Model.f_train(ds_train_ready, ds_val_ready)

############ Results ############
results = Audio_Model.model.evaluate(ds_test_ready)

print(f"Loss : {results[0]}")
print(f"Accuracy : {results[1] * 100:.2f}%")

## Plotting losses and accuracy
plt.figure()
plt.title("Accuracy")
plt.xlabel('epoch')
plt.plot(Audio_Model.training_accuracy, '--', color='blue')
plt.plot(Audio_Model.validation_accuracy, '-', color='blue')
plt.legend(['Training accuracy', 'Validation accuracy'])
#plt.ylim(0, 1.2)

plt.figure()
plt.title("Losses")
plt.xlabel('epoch')
plt.plot(Audio_Model.training_loss,'--', color='orange')
plt.plot(Audio_Model.validation_loss, '-', color='orange')
plt.legend(['Training loss', 'Validation loss'])
#plt.ylim(0, 2)

############ Confusion Matrix ############
y_pred = []
y_true = []
for audio, label in ds_val_ready:
    y_true.extend(label.numpy())
    
    prediction = Audio_Model.model.predict(audio, verbose = 0)
    
    y_pred.extend(np.argmax(prediction, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

cm = sklearn.metrics.confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', vmax = 1000)

plt.xlabel('Predicated Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Audio Recognition')
plt.show()

## Score
Report = sklearn.metrics.classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn')
print(Report)

plt.figure(figsize=(6, 4))
plt.suptitle("Mel Spectrogram")
i = 0
for audio, label in ds_val.take(25):
    audio_numpy = audio.numpy()
        
    volume=0.5

    prepro = preprocess(audio_numpy, label)[0]
    spectrogram = Spectrogram(prepro, label)[0]
    input_tensor = np.expand_dims(spectrogram, axis=0)
    prediction = Audio_Model.model.predict(input_tensor, verbose = 0)
    indice = np.argmax(prediction)

    print(f"True Label ID : {label} ({label_name[label]})")
    print(f"Label Prediction : {label_name[indice]}")
    print('\n')
    
    ax = plt.subplot(5, 5, i + 1)        
    ax.imshow(input_tensor.squeeze().T, aspect='auto', origin='lower', cmap='viridis')
    ax.axis('off')
    ax.set_title(label_name[label], fontsize=10)
    i+=1
plt.tight_layout()

plt.show()            

## Saving
save=0
if save:
    fil_name = "Model_Audio"
    Audio_Model.model.save(fil_name+'.keras')

    attributes = {
        'training_accuracy': Audio_Model.training_accuracy,
        'training_loss': Audio_Model.training_loss,
        'validation_accuracy': Audio_Model.validation_accuracy,
        'validation_loss': Audio_Model.validation_loss,
    }
    with open(fil_name+'.pkl', 'wb') as f:
        pickle.dump(attributes, f)
        
    print("Model saved")
    
    
################################################
## Prediction the gender based on a .wav file 
################################################

def predict_audio(file_path, file):
    sample_rate = 16000
    mon_audio, _ = librosa.load(file_path + file, sr=sample_rate)
    #mon_audio, _ = librosa.effects.trim(mon_audio, top_db=5)
    mon_audio = tf.convert_to_tensor(mon_audio, dtype=tf.float32)  
    
    nb = int(np.ceil( len(mon_audio)/(sample_rate*3) ) )
    zero_padding = tf.zeros([sample_rate*3*nb] - tf.shape(mon_audio), dtype=tf.float32)
    audio_dec = tf.concat([mon_audio, zero_padding], axis=0)

    audio_dec = tf.reshape(audio_dec, [nb,sample_rate*3,1])

    audio_np = np.array(mon_audio)
    sd.play(audio_np[0:48000*3]*0.1/np.max(np.abs(audio_np[0:48000])), 16000)

    ds_segments = tf.data.Dataset.from_tensor_slices(audio_dec)
    ds_segments = ds_segments.map(lambda x: preprocess(x, 0)[0]) \
        .map(lambda x: Spectrogram(x, 0)[0]) \
        .batch(128)

    predictions = Audio_Model.model.predict(ds_segments)
    Final_prediction = label_name[np.argmax(np.mean(predictions, axis=0))]

    print(f"A {Final_prediction} is speaking")

    
    
file_path = "..." # Path to the audio file
file = "Name.wav" # Name of the audio file

predict_audio(file_path, file)
