"""
python3.12 Scripts/train_conv_lstm_model.py -in_dir Dataset -out_dir Models
"""
import numpy as np
import os
import argparse
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Bidirectional, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical
from keras.optimizers import Adam

def scaleData(trainX, testX):
    
    sc = StandardScaler()
    
    trainX = sc.fit_transform(trainX.reshape(-1, trainX.shape[-1])).reshape(trainX.shape)
    testX = sc.transform(testX.reshape(-1, testX.shape[-1])).reshape(testX.shape)
    
    return trainX, testX


def buildModel(inShape, numClasses):
    
    model = Sequential()
    
    model.add(Input(shape=inShape))
    
    model.add(Conv1D(32, kernel_size=7, activation='relu')) 
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    
    model.add(Conv1D(64, kernel_size=5, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    
    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    
    model.add( Bidirectional(LSTM(64)) )
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(numClasses, activation='softmax'))
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    return model




parser = argparse.ArgumentParser()
parser.add_argument("-in_dir", required=True)
parser.add_argument("-out_dir", required=True)
args = parser.parse_args()

inPath = os.path.join(os.getcwd(), args.in_dir)
outPath = os.path.join(os.getcwd(), args.out_dir)
os.makedirs(outPath, exist_ok=True)

X = np.load(os.path.join(inPath, "X.npy"))
y = np.load(os.path.join(inPath, "y.npy"))
groups = np.load(os.path.join(inPath, "groups.npy"))

le = LabelEncoder()
yEnc = le.fit_transform(y)
numClasses = len(le.classes_)

np.save(os.path.join(outPath, "classes.npy"), le.classes_)

logo = LeaveOneGroupOut()

allTrue = []
allPred = []

fold = 1

for trainIdx, testIdx in logo.split(X, yEnc, groups):
    
    testPart = groups[testIdx][0]
    print(f"\nProcessing Fold ### {fold} ### - Testing on {testPart}")

    trainX = X[trainIdx]
    testX = X[testIdx]
    trainY = yEnc[trainIdx]
    testY = yEnc[testIdx]

    sd = scaleData(trainX, testX)
    trainX = sd[0]
    testX = sd[1]

    trainYCat = to_categorical(trainY, numClasses)
    testYCat = to_categorical(testY, numClasses)

    weightsArr = compute_class_weight('balanced', classes=np.unique(trainY), y=trainY)
    
    MAX_WEIGHT = 5.0
    classWeights = {}
    
    for cls, w in zip(np.unique(trainY), weightsArr):
        classWeights[cls] = min(w, MAX_WEIGHT)
        
    for i in range(numClasses):
        if i not in classWeights:
            classWeights[i] = 1.0

    model = buildModel((X.shape[1], X.shape[2]), numClasses)
    
    savePath = os.path.join(outPath, f"lstm_model_{fold}.keras")
    
    cbs = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(savePath, monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    ]

    model.fit(
        trainX, trainYCat,
        validation_data=(testX, testYCat),
        epochs=60,
        batch_size=64,
        class_weight=classWeights,
        callbacks=cbs,
        verbose=1
    )

    predProb = model.predict(testX)
    preds = np.argmax(predProb, axis=1)

    allTrue.extend(testY)
    allPred.extend(preds)

    fold += 1

print("\n### Final conv-LSTM LOPOCV Results ###")

acc = accuracy_score(allTrue, allPred)
prec = precision_score(allTrue, allPred, average='weighted')
rec = recall_score(allTrue, allPred, average='weighted')

print(f"Accuracy:  {acc}")
print(f"Precision: {prec}")
print(f"Recall:    {rec}")

cm = confusion_matrix(allTrue, allPred)
print("\nConfusion Matrix:")
print(cm)

np.save(os.path.join(outPath, "cm_lstm.npy"), cm)

with open(os.path.join(outPath, "report.txt"), "w") as f:
    f.write(classification_report(allTrue, allPred, target_names=le.classes_,zero_division=0))