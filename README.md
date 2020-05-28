# LoRD
Logistic Regression Design for QCD taggers at the LHC

These codes were used to obtain the results in this paper:
"Jet tagging made easy"  (https://inspirehep.net/literature/1782677, accepted for publication in European Physical Journal C)

They consist of a Logistic Regression implementation, using Python_3 and tensorflow_1.4.0 for the optimization procedure. 

Data consists of 23 atttributes, corresponding to different elements of the "subjettiness basis" (cf ref. above), classified as signal or background (binary classification). There are different types of signals and backgrounds, all merged into 2 classes. 

Please note that this repository contains just a sample of the data used in the paper (containing only 100 events per type of signal or background), for the sake of checking the code's working flow. The results shown in the paper are obtained with a much larger sample of events for each type of process. 

Content of the repository:
- "LR_4Ptagger_std1000_M80.py":  this is the code for the four-pronged tagger
- "LR_3Ptagger_std1000_M80.py":  this is the code for the three-pronged tagger
- "LR_2Ptagger_std1000_M80.py":  this is the code for the two-pronged tagger
- "train_data": this is the directory with train data, both for signal and background
- "test_data": this is the directory with the test data (signal)

