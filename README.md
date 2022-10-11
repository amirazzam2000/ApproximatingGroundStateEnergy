# Aproximating the Ground State of The Molecular Hamiltonian 

A project aiming  at approximating the ground state energy of the molecular Hamiltonian using machine learning and quantum computing.

The full description of the project can be found in the report file "_Final_Thesis___Amir_Azzam.pdf_".

The notebook _Generating_data_ allows you to generate more data given a new moleucle. This data is saved in a text file in the Text folder. 

The notebook _translating_data_ allows you to translate the files you generated from a text file into a pickle that can be used to train the model. 

The notebook _Model_ allows you to train the model with the data you have constructed by specifying the files that need to be included in the training process. 

The file _LSTM_Model_ contains the code to the Long Short Term Memory used by this project. 

The file _hamiltonian_functions_ contains the code needed to find the ground state energy of a molecule on a quantum computer. 
