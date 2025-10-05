import pickle

# Load and display first pickle file
print("Contents of lab1_data_2s.pkl:")
with open('lab1_data_2s.pkl', 'rb') as f:
    data_2s = pickle.load(f)
print(data_2s)
print("\n" + "="*50 + "\n")

# Load and display second pickle file
print("Contents of lab1_data_10s.pkl:")
with open('lab1_data_10s.pkl', 'rb') as f:
    data_10s = pickle.load(f)
print(data_10s)