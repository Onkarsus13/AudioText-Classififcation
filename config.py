import torch

batch_size = 20
max_len = 10
num_worker = 2
shuffle = False


train_path = 'task_data/train_data.csv'
val_path = 'task_data/valid_data.csv'
test_path = '' #need test.csv 
model_path = 'saved_model/'


lr = 0.0001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 100

