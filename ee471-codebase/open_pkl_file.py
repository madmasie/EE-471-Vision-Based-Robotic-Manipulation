import pickle

file_path = 'lab3_3_data.pkl'  
with open(file_path, 'rb') as file:
    data = pickle.load(file)
    print(data)