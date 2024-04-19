# %%
import json
import pandas as pd
from numpy import array

import torch
import torch.nn
import torch.optim as optim

# Used in LTSMModel Class Instantiation
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %%
def ConstantMinMaxScaler(series,min_gold,max_gold):
    dataset_scaled = (series - min_gold)/(max_gold - min_gold)
    return dataset_scaled

def ConstantUnScaler(series, min_gold, max_gold):
    dataset_unscaled = np.round((series * (max_gold - min_gold)) + min_gold)
    return dataset_unscaled

# %%
%run data_prep_one_hero.ipynb

#returns df_allhero with dimensions (124, 3)

# %% [markdown]
# ### Dataset Class

# %%
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    # Class to create our dataset
    def __init__(self, df):
        self.hero_ids = df['hero_id'].values # Declaring hero_id values
        self.time_series = [torch.tensor(ts) for ts in df['gold_t']] # Converting the time_series into Tensors
        self.max_length = max(len(ts) for ts in self.time_series) # Grabs max length of all the tensors to pad them with 0s later
        self.match_ids = df['match_id'] #Storing the match_id in case we want to view this later for more info


    def __len__(self):
        return len(self.hero_ids) # Convenient length call
    

    def __getitem__(self, idx):

        hero_id = self.hero_ids[idx]
        time_series = self.time_series[idx]
        match_id = self.match_ids[idx]

        scaled_time_series = ConstantMinMaxScaler(time_series, min_gold, max_gold)
        
        length = len(scaled_time_series)

        padded_time_series = torch.zeros(self.max_length)
        padded_time_series[:length] = scaled_time_series
        #print(f"Padded Time Series: {padded_time_series}")

        return hero_id, padded_time_series, length

# %% [markdown]
# ### Embedding Layer

# %%
import torch
import torch.nn as nn

class ProcessEmbedding(nn.Module):
    def __init__(self, df, embedding_dim):
        super(ProcessEmbedding, self).__init__() 

        self.num_processes = len(df['hero_id'].unique()) # declaring number of different categories of time-series for dimensionialty reasons
        self.embedding_dim = embedding_dim # passing our embed size to be a class attribute
        self.process_embeddings = nn.Embedding(self.num_processes, embedding_dim)

        self.hero_id_to_idx = {hero_id: idx for idx, hero_id in enumerate(df['hero_id'].unique())}

    def forward(self, hero_ids):
        process_ids = [self.hero_id_to_idx[hero_id.item()] for hero_id in hero_ids]
        process_ids = torch.tensor(process_ids)
        process_embeddings = self.process_embeddings(process_ids)

        print("Process Embeddings shape:", process_embeddings.shape)
        print("Process Embeddings tensor:", process_embeddings)

        return process_embeddings

# %% [markdown]
# ### LSTM Class

# %%
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, process_embedding):
        super(LSTMModel, self).__init__() # ensures the correcty PyTorch class is also initialized

        self.hidden_size = hidden_size #hyper param 
        self.num_layers = num_layers #hyper param

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) # Actual LSTM creation
        self.fc = nn.Linear(hidden_size, output_size) # Linear Model creation
        self.process_embedding = process_embedding # Process Embedding


    def forward(self, hero_ids, time_series):
        batch_size = time_series.size(0) # pulling dims from the tensor
        seq_length = time_series.size(1) # pulling dims from the tensor
        
        # Get process embeddings for hero_ids
        process_embeddings = self.process_embedding(hero_ids)

        print("Process Embeddings shape:", process_embeddings.shape)
        print("Time Series shape:", time_series.shape)
        
        # Reshape process embeddings to match the input shape of LSTM
        process_embeddings = process_embeddings.unsqueeze(1).repeat(1, seq_length, 1)

        print("Reshaped Process Embeddings shape:", process_embeddings.shape)

        # Unsqueexing to ensure the time_series shape is 3D like our embedding processing is so that no issues are ran into with torch.cat below
        time_series = time_series.unsqueeze(-1)

        print("Time Series shape with extra dimension:", time_series.shape)
        
        # Concatenate process embeddings with time series data
        # dim = -1, signifies concatenation across the last dimension (the feature dimension)
        input_data = torch.cat((process_embeddings, time_series), dim=-1)
        
        # Pack the padded sequences
        # Packing the padded Sequences is a way of optimizing computation times. We have padded the time series to all be the same length, even though some are only 20 or less
        # The packing indicates which are the real values in the time series so that the computation is only ran on those time steps. Details on how are unknown to me thus far.
        packed_input = pack_padded_sequence(input_data, lengths, batch_first=True, enforce_sorted=False)


        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        

        packed_output, _ = self.lstm(packed_input, (h0, c0))

        # Unpack the output
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Take the last output of the LSTM
        out = self.fc(output[:, -1, :])
        
        return out

# %% [markdown]
# ### Instantiating Classes and Parameters

# %%
process_embedding = ProcessEmbedding(df_allhero, embedding_dim=84) # we create the embedding vector on unsplit data to ensure all unique hero id's are contained

#input_size = process_embedding.embedding_dim + time_series.shape[-1]  # Number of features (embedding_dim + time_series_dim)
input_size = process_embedding.embedding_dim + 1 #84 + 1
hidden_size = 64
num_layers = 2
output_size = 1  # Assuming you want to predict a single value


model = LSTMModel(input_size, hidden_size, num_layers, output_size, process_embedding)

# %%
# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
batch_size = 32

# %%
test_size = .30

train_df, test_df = train_test_split(df_full, test_size=test_size, shuffle=False)

# %%
train_dataset = TimeSeriesDataset(train_df)
test_dataset = TimeSeriesDataset(test_df)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%
# Training loop
num_epochs = 1000
batch_size = 32

for epoch in range(num_epochs):
    model.train()
    
    train_loss = 0.0
    print(f"Epoch: {epoch}")
    for hero_ids, time_series, lengths in train_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(hero_ids, time_series)
        targets = time_series[:, -1]  # Assuming you want to predict the last value of each time series
        loss = criterion(outputs.squeeze(), targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * time_series.size(0)

    train_loss /= len(train_dataset)

    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for hero_ids, time_series, lengths in test_loader:
            # Forward pass
            
            outputs = model(hero_ids, time_series)
            targets = time_series[:, -1]  # Assuming you want to predict the last value of each time series
            loss = criterion(outputs.squeeze(), targets)

            test_loss += loss.item() * time_series.size(0)

    test_loss /= len(test_dataset)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")


