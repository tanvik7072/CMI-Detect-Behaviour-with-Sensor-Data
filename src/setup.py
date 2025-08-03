
#data input
raw_train_path = r'data/train.csv' 
train_demographics_path = r'data/train_demographics.csv'
raw_test_path = r'data/test.csv'
test_demographics_path = r'data/test_demographics.csv'

#output paths
output_predictions_path = r'output/predictions.csv' 
best_model_path = r'output/best_model.pth' 
end_model_path = r'output/end_model.pth' 

#init hyperparameters
batch_size = 64 
num_epochs = 100
learning_rate = 0.001 

#init device
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #use GPU if available, else CPU

