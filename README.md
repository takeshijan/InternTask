# InternTask
## Manbaa havolalari

- ### 🔗 Dataset  ([Google Drive](https://drive.google.com/drive/folders/1OePubp1rEFIALGG668Fo-74OX2WyS5Ei?usp=share_link)) 

- ### 🔗 Kaggle uchun [havola](https://www.kaggle.com/code/nodirjonmuxammadali/intern)

## Texnologiyalar
- Tensorflow (Keras)
- Pandas
- Numpy
## Qo'shimcha kerakli kutubxonalar
- `pip install jiwer`
- `pip install pydub`

#

# Modelni ishga tushurish
⚠️ Dataset kamligi va o'qitish epochlar soni kamligi sababli model aniqligi yuqori emas !
⚠️ Modelni ishga tushurishdan avval GPU yoqilganligiga ishonch hosil qiling

⚠️ **model.h5** faylini Google Colab ga yuklab oling. Yuklash qulay bo'lishi uchun [havola](https://drive.google.com/file/d/1d0T4VmlnuwXw647jBBKXTmxQpRtR1EVV/view?usp=share_link)

```python
import keras
import tensorflow as tf
import numpy as np 


reloaded_model = keras.models.load_model("model.h5", compile=False)

frame_length = 256
frame_step = 160
fft_length = 384

characters = [x for x in """abcdefghijklmnopqrstuvwxyz' """]
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)


def decoder(pred):
    
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text


def preprocess_audio(audio_file_path):
    file = tf.io.read_file(audio_file_path)
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    audio = tf.cast(audio, tf.float32)
    spectrogram = tf.signal.stft(
        audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
    )
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)
    return spectrogram


def predict_transcription(audio_file_path):
    spectrogram = preprocess_audio(audio_file_path)
    prediction = reloaded_model.predict(tf.expand_dims(spectrogram, 0))
    transcription = decoder(prediction)[0]
    return transcription


wav_file_path = "audio.wav" #Audio fayl pathini belgilang
transcription = predict_transcription(wav_file_path)
print("Predicted :", transcription)

```

