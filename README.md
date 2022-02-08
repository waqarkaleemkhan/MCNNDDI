# MCNNDDI
(MCNNDDI) Multimodal Convolutional Neural Network for drug drug associated events predictions.
Used Drug bank dataset.
And achived an accuracy of 90.00
The feature of drugs are used smiles (Chemical substructure), target,
Enzyme, and pathways and Drug Interactions.
The events.db file contain data of the paper which is a sqlite file.
For extraction of data use spider.py file.
For doing different experiments change the model type like (RF, DDIMDL, CNN, LSTM, RNN, etc) in main function and also change the features of drugs like enzymes, targets, pathways etc or combination of features enzymes + target, smiles + target, smiles + Enzyme, smiles+target+pathways. The best result are given by smiles+target+pathways these features. And Smiles show best results alone. And the pathways show low results individualy. That's why the above three features give high performance on MCNN. 



