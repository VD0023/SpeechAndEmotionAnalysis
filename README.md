# Speech and Emotion Analysis using Python

## Introduction
Speech and emotion analysis is a fascinating field that involves analyzing 
and extracting insights from spoken language and detecting emotions expressed in 
the speech. This project aims to provide a practical implementation of speech and 
emotion analysis using Python. The project consists of three main components: 
Emotion Analysis, Speech Recognition, and Voice Recording & Recognition.

![image](https://github.com/VD0023/SpeechAndEmotionAnalysis/assets/99820386/a6b08c60-cb64-4232-bbf8-77020343c495)


## Emotion Analysis
The Emotion Analysis component focuses on analyzing text data to determine the 
emotions expressed in the given text. The code utilizes the Python libraries such 
as numpy, pandas, nltk, and scikit-learn to perform text preprocessing, feature 
extraction, and emotion classification. It employs a Random Forest classifier 
along with TF-IDF vectorization to train a model on a dataset containing labeled 
emotions. The accuracy of the model is evaluated using the test set and a confusion
matrix is generated for visualization.

## Speech Recognition
The Speech Recognition component involves recognizing spoken words or phrases from
audio data. The code utilizes the librosa library for audio processing, including 
loading audio files, resampling the audio to a specific sample rate, and extracting 
audio features. It applies a Convolutional Neural Network (CNN) architecture 
implemented with Keras to classify the audio samples into different predefined
classes. The model is trained on a labeled dataset containing audio samples of 
various words or phrases. The training and testing sets are split, and the accuracy
of the model is evaluated.

![image](https://github.com/VD0023/SpeechAndEmotionAnalysis/assets/99820386/c0047188-676a-4b0a-ad5d-13c45301e8f2)

## Voice Recording & Recognition
The Voice Recording & Recognition component allows users to record their own voice 
samples and recognize the spoken word using the trained speech recognition model. 
The code integrates the sounddevice library to record audio from the microphone, 
saves the recorded audio as a WAV file, and then applies audio preprocessing techniques
similar to the Speech Recognition component. The saved audio file is loaded, resampled, 
and fed into the trained model for prediction. The predicted class is returned based on 
the highest probability from the model's output.

## Usage
To use this project for speech and emotion analysis, follow the steps below:

1. Ensure that the required Python libraries are installed, including 
2. numpy, pandas, nltk, scikit-learn, librosa, Keras, sounddevice, and soundfile.

2. Prepare the necessary datasets:
   - For Emotion Analysis, provide a CSV file containing labeled text data,
      where each row represents a text sample along with its corresponding emotion label.
   - For Speech Recognition, organize a dataset containing audio files for each
      class you want to recognize. Ensure the audio files are properly labeled.

3. Execute the code for each component:
   - Emotion Analysis: Run the code for text preprocessing, feature extraction, 
     training a Random Forest classifier, and evaluating the model's accuracy.
   - Speech Recognition: Run the code for audio processing, training a CNN model,
     and evaluating the model's accuracy.
   - Voice Recording & Recognition: Execute the code for recording voice samples, 
     loading the trained model, and predicting the spoken word.

4. Interpret and analyze the results:
   - For Emotion Analysis, analyze the accuracy of the model and interpret the 
     confusion matrix to understand the performance of different emotion classifications.
   - For Speech Recognition and Voice Recognition, evaluate the accuracy of the
     models by comparing the predicted classes with the ground truth labels. Explore 
     the potential applications and implications of the speech recognition system.

![image](https://github.com/VD0023/SpeechAndEmotionAnalysis/assets/99820386/15a239e2-36d2-4520-a3f2-fc14641c6328)


## Conclusion
The Speech and Emotion Analysis project provides a comprehensive implementation of 
analyzing speech data and detecting emotions using Python. It combines text analysis 
techniques, audio processing, and machine learning models to achieve accurate results.
This project can be extended further by exploring advanced techniques such as deep learning
architectures, sentiment analysis, or integrating other features like speaker recognition.
The applications of speech and emotion analysis are wide-ranging,

 including sentiment analysis in social media, customer feedback analysis, and voice-controlled systems.

Please note that this documentation serves as a high-level overview of the project
and does not include every detail or code explanation. Refer to the individual code sections
for more in-depth information and comments.

*Disclaimer: This project documentation is provided as a reference and guide. Please ensure
that you comply with all necessary legal and ethical requirements when working with speech data
, including obtaining appropriate consent and protecting privacy rights.*
