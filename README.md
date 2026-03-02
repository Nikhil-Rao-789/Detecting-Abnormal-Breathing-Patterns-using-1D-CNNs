# Detecting-Abnormal-Breathing-Patterns-using-1D-CNNs
## AI for Healthcare
This project focuses on detecting abnormal breathing patterns by feeding time series data such as Nasal Flow, Thoracic movements and SPO₂ into 1D CNNs. It also explores the effect of changing parameters such as Learning Rates and number of epochs on Accuracy and Precision.

### Repository Structure
- Data : This directory contained raw data of the participants in form of .txt files - "SPO2.txt", "Flow.txt", "Flow Events.txt", "Sleep profile.txt" , "Thorac.txt" for each participant (data is not kept in the repository)
- Dataset : This directory contained the processed data ready for training models in form of .npy files - "groups.npy", "stages.npy", "X.npy", "y.npy" (dataset is not kept in the repository)
- Scripts : This directory contains the following .py files - "vis.py" for creating the visualizations, "create_dataset.py" for creating the final refined dataset, "train_model.py" to train Model with a GlobalAveragePooling1D layer, "train_conv_lstm_model.py" to train Model with Bidirectional(LSTM(64)) layer. It also contains a package "myPackage" containing some utility modules like "createDfs.py" and "cleanDfs.py".
- Visualizations : This directory contains sample visualization pdfs for 3 participants.
- CNN Models : This directory contains the models (.keras files), confusion matrix ("cm.npy") and a "report.txt" file containing parameters like accuracy and precision, generated while using "train_model.py" (note - the code saves the models in a generic directory called "Models")
- CNN-LSTM models : This directory contains the models (.keras files), confusion matrix ("cm_lstm.npy") and a "report.txt" file containing parameters like accuracy and precision, generated while using "train_conv_lstm_model.py" (note - the code saves the models in a generic directory called "Models")
- images : This directory contains images used in the readme.md file
- report.pdf : This file contains the gist of this project

### Phases in the Project -
### Phase 1 : Data Cleaning and Visualization
The image below shows the general structure of the data of individual participants.

![General Structure of data in each Participant](images/AP02-data-folder.png)

This data is loaded, cleaned and then plotted for visualization.

The vis.py file finishes its execution-

![General Structure of data in each Participant](images/Generating-visualizations.png)

Sample of the graphs generated-

![General Structure of data in each Participant](images/visualizations.png)

### Phase 2 : Dataset Creation
Once the data is visualized and understood, it is converted into a dataset usable for ML Models.
The create_dataset.py file finishes execution-

![General Structure of data in each Participant](images/dataset-creation.png)

Once done, the program saves the dataset in the format given below-

![General Structure of data in each Participant](images/dataset.png)

Now, we are ready to train our models on this dataset.

### Phase 3 : Model Training and analysis

The Model is trained- 

![General Structure of data in each Participant](images/model-training.png)

I trained multiple models tweaking certain parameters here and there while analyzing the outputs

The below image shows the report of the cnn model with the GlobalAveragePooling1D layer

![General Structure of data in each Participant](images/cnn-model-report.png)

The below image shows the report of the cnn model with the Bidirectional(LSTM(64)) layer. Perhaps using a bigger dataset might have improved the results.

![General Structure of data in each Participant](images/cnn-lstm-model-report.png)

While most of the code was written by me, I took counsel of gemini and chatgpt to learn how to use libraries that were new to me such as tqdm for a loading bar and for braistorming ways to increase accuracy and prescision of the models

### What I Learnt -
I got hands on experience dealing with raw IoMT/IoHT sensor data, how to clean that data, how to visualize that data, how to turn that data into a dataset useful for ML and training CNN models for time series data.  
This Project taught me that cleaning data plays a vital role in model accuracy (had first hand experience of mishaps)
