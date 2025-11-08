# Birdcall Classifier Project

## What is it? 
This classifier categorizes birdcalls from a labelled dataset of four bird species. With the input of a .wav file containing a potential birdcall, the classifier builds a decision tree from the training set and outputs predictions from a test set to a csv file with approximately 80% accuracy. 

## Why?
Feature extraction is a vital part of data science and machine learning workflows. To practice feature extraction that can effectively classify bird calls, I limited myself to only classical machine learning methods (kNN, SVM etc) to challenge myself to carefully consider feature extraction. 

## Dataset
The dataset consists of 3 second audio samples extracted from longer recordings, featuring a variety of bird species and environmental conditions. Participants must preprocess and extract valuable features from these samples to create a classifier model. These features must be robust to noise/variability in recording quality while navigating challenges like multiple overlapping, similar sounding bird calls. The dataset is copyright of Jayadev Acharya, a professor at Cornell, and used with permission. 

![](https://upload.wikimedia.org/wikipedia/commons/0/03/201903_Zebra_finch.svg)
<img src="https://upload.wikimedia.org/wikipedia/commons/0/03/201903_Zebra_finch.svg" width="48">

## Data Processing/Feature Extraction:
Feature extraction was done via the Librosa package. Librosa is a package for audio analysis, and thus applicable to the task at hand. I added features in the following order to achieve the maximum possible accuracy. The features were flattened and concatenated into a 1d array for model training in the following order.

### Librosa.feature.mfcc:
Mel-frequency cepstral coefficients (MFCCs) are features widely used in audio analysis and speech recognition to represent the spectral properties of a sound signal in a way that mimics human auditory perception. MFCCs are computed using the Mel scale, which is a nonlinear scale that approximates the way humans perceive pitch and frequency. On the Mel scale, lower frequencies are more finely resolved than higher frequencies. They reduce the high-dimensional raw audio signal into a low-dimensional feature vector.

### Librosa.feature.rms: 
Root-mean-square (RMS) values are another statistical measure of center in audio analysis. RMS values provide a way to summarize the overall amplitude of a signal. As the name suggests, they are calculated by taking the square root of the mean of squares of values. Unlike other spectral features, RMS values are not calculated from the FFT. 

### librosa.feature.spectral_centroid: 

This feature characterizes the “center of mass” of a frequency spectrum. The spectral centroid is computed as the weighted mean of the frequencies present in the sound, where the weights are the magnitudes of the frequencies in the spectrum.  A violin playing high notes will have a higher spectral centroid compared to a bass guitar playing low notes.

### librosa.feature.spectral_contrast: 
Feature that describes the difference in energy levels between high-energy frequencies and low-energy frequencies of a frequency spectrum. Spectral contrast is similar to spectral bandwidth and spectral centroid, however it particularly emphasizes the dynamic range within frequency bands.

### librosa.feature.spectral_bandwith:
Measure of the spread of frequencies in an audio signal around a central frequency, often the spectral centroid. Because it measures spread, calculation of spectral bandwidth is similar to standard deviation.

### librosa.feature.spectral_flatness:
Feature used to describe how "flat" or "peaky" the frequency spectrum of a sound is. A pure sine wave has most of its energy concentrated at a single frequency, so the spectral flatness is low (close to 0). Meanwhile, white noise has equal energy across all frequencies, so the spectral flatness is high (close to 1). Spectral flatness is particularly useful in distinguishing between tonal and noise-like sounds.

### librosa.feature.spectral_rolloff:
Feature used to describe the frequency below which a specified percentage (typically 85% or 95%) of the total spectral energy is concentrated. It is a way of quantifying how the energy in an audio signal is distributed across the frequency spectrum.

### Mel spectrogram: 
I tested models with raw flattened mel spectrogram input. However, this actually decreased the accuracy of the models. For instance, the accuracy of gradient boosting decreased from 70% to 50% when the raw Mel spectrogram was included. The data included in the raw mel spectrogram was too high a dimensionality and too redundant with the other features to be worth it. 

## Model Selection:
I compared three models: k-NN, SVM, and gradient boosting. I found that I could get a 10-20% increase using gradient boosting over k-NN and SVM. Additionally, given that sklearn has an efficient implementation of gradient boosting using histograms it was not much more computationally costly compared to the other models. Finally, independent of this class, I am interested in gradient boosting because it plays a significant role in my undergraduate research in a biomedical engineering lab, where we use it to classify spectroscopy data collected from blood cells.

Summary of Gradient Boosting:

The basic unit in modern gradient boosting algorithms is a decision tree. A decision tree is a simple model for classification. It resembles a human made decision tree, such as a flowchart. The tree is made via recursive partitioning. Recursive partitioning can be approached through several mathematical strategies, but it essentially just aims to minimize the amount of misclassified data for a given feature. One way to do this is with Gini Impurity, which quantifies how “impure” a partition is. It is one minus the sum of the frequency of each class squared, so 0 is perfectly pure, and an even mix of four classes would be .75, which is impure. A decision tree will take the feature that best splits the data and split the whole data set (or root node) into two parts. Then, the resulting two nodes are split again, continuing recursively to a certain endpoint defined by the hyperparameters. At the end of each node is a “leaf”, or the best classification for the data in the node. 

![A simple decision tree. Source: Wikipedia](https://upload.wikimedia.org/wikipedia/commons/e/eb/Decision_Tree.jpg)

Gradient boosting is an ensemble approach, meaning that it combines many less accurate decision trees into a more powerful model. The algorithm to do so is called “boosting”, hence its name. Boosting in classification is based on making a new tree on the gradient of predicted log-odds, which is a measure of the residuals (errors) of the previous attempt. The update to the new tree is scaled by a hyperparameter called shrinkage. When using gradient boosting algorithms, one tradeoff is scaling this shrinkage, in addition to the number of trees the model iterates through. A high shrinkage allows the model to be trained with fewer iterations; however, it increases the chance of overfitting the data. Meanwhile, the “careful” approach of a low shrinkage and many estimators is more computationally intensive, but produces a more accurate model.

Tuning shrinkage and the number of estimators was difficult due to the size of the dataset and number of features being inputted into the model. I settled on a shrinkage rate of .01 and a maximum number of estimators of 1500. These hyperparameters increased the accuracy by 2% while still keeping the program’s total runtime to under an hour.
