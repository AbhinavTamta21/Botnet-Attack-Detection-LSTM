# Botnet-Attack-Detection-LSTM
# Botnet Detection Using a Hybrid Deep Learning Model

## Overview

This project focuses on detecting Botnet attacks using a hybrid deep learning model that combines Deep Neural Network (DNN) and Long Short-Term Memory (LSTM) layers. The goal is to classify network traffic and identify potential Botnet activities.

## Table of Contents

1. [Dataset Description](#dataset-description)
2. [Solution Notebook Summary](#solution-notebook-summary)
3. [Importing Libraries](#importing-libraries)
4. [Load and Preprocess Data](#load-and-preprocess-data)
5. [Build the DNN Model](#build-the-dnn-model)
6. [Build the LSTM Model](#build-the-lstm-model)
7. [Combine DNN and LSTM Models](#combine-dnn-and-lstm-models)
8. [Compile the Model](#compile-the-model)
9. [Train the Model](#train-the-model)
10. [Evaluate the Model](#evaluate-the-model)
11. [Output Video](#output-video)

## Dataset Description

The dataset consists of the following features:

1. **ID**: Unique identifier for each record.
2. **Sender_IP**: IP address of the sender.
3. **Sender_Port**: Port number on the sender's side.
4. **Target_IP**: IP address of the target.
5. **Target_Port**: Port number on the target's side.
6. **Transport_Protocol**: Protocol used for communication (e.g., TCP, UDP).
7. **Duration**: Duration of the communication.
8. **AvgDuration**: Average duration of multiple connections.
9. **PBS (Payload Byte Size)**: Size of the payload in bytes.
10. **AvgPBS**: Average payload byte size over multiple connections.
11. **TBS (Total Byte Size)**: Total size of the transmitted data.
12. **PBR (Payload Byte Rate)**: Rate of payload bytes transmitted per unit time.
13. **AvgPBR**: Average payload byte rate over multiple connections.
14. **TBR (Total Byte Rate)**: Total byte rate including payload and headers.
15. **Missed_Bytes**: Number of bytes not successfully transmitted or received.
16. **Packets_Sent**: Total number of packets sent.
17. **Packets_Received**: Total number of packets received.
18. **SRPR (Sender-to-Receiver Packet Ratio)**: Ratio of packets sent to packets received.
19. **class**: Label indicating whether the communication is a Botnet attack (1) or normal (0).

## Solution Notebook Summary

The provided code demonstrates the following steps:

1. **New Data Preparation**: Create and format new data as a pandas DataFrame.
2. **Scaling New Data**: Apply the same scaler used during training to the new data.
3. **Reshaping for LSTM**: Reshape new data to match the input shape expected by the LSTM layer.
4. **Making Predictions**: Use the trained hybrid model to predict the likelihood of a Botnet attack.
5. **Displaying Results**: Print the probability and binary prediction.

## Importing Libraries

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
```

## Load and Preprocess Data

```python
# Load dataset
data = pd.read_csv('dataset.csv')

# Separate features and target
X = data.drop('class', axis=1)
y = data['class']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

## Build the DNN Model

```python
input_layer = Input(shape=(X_train.shape[1],))
dense_layer = Dense(64, activation='relu')(input_layer)
output_layer_dnn = Dense(1, activation='sigmoid')(dense_layer)

dnn_model = Model(inputs=input_layer, outputs=output_layer_dnn)
```

## Build the LSTM Model

```python
input_layer_lstm = Input(shape=(X_train.shape[1], 1))
lstm_layer = LSTM(64)(input_layer_lstm)
output_layer_lstm = Dense(1, activation='sigmoid')(lstm_layer)

lstm_model = Model(inputs=input_layer_lstm, outputs=output_layer_lstm)
```

## Combine DNN and LSTM Models

```python
combined_output = concatenate([output_layer_dnn, output_layer_lstm])
final_output = Dense(1, activation='sigmoid')(combined_output)

hybrid_model = Model(inputs=[input_layer, input_layer_lstm], outputs=final_output)
```

## Compile the Model

```python
hybrid_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
```

## Train the Model

```python
X_train_lstm = np.expand_dims(X_train, axis=-1)
X_test_lstm = np.expand_dims(X_test, axis=-1)

early_stopping = EarlyStopping(patience=5)

hybrid_model.fit([X_train, X_train_lstm], y_train, epochs=50, batch_size=32, validation_data=([X_test, X_test_lstm], y_test), callbacks=[early_stopping])
```

## Evaluate the Model

```python
evaluation_results = hybrid_model.evaluate([X_test, X_test_lstm], y_test)
print(f'Test Accuracy: {evaluation_results[1] * 100:.2f}%')
```

## Output Video

For a visual demonstration, refer to the [output video](https://drive.google.com/file/d/1BNpPf6v1khUGJjHq1K5dAAdyHcpA4FZt/view?usp=sharing).

---

