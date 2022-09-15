Where does the actual data come from? 
==========

Equistore manages the metadata, where does the data come from. How does equistore deal with it and how to register new data origins in the python wrapper 

(Python automagically transforms data to numpy.ndarray or a torch.tensor)  
Equistore has no idea about the data itself (only knows the pointer to data and operations you can perform on it - create, destroy move and reshape data)


