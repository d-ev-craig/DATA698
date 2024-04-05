# %%
import json
import pandas as pd
from numpy import array

import torch
import torch.nn
import torch.optim as optim

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %%
file_path = "C:\\Users\\dcrai\\source\\repos\\DATA698\\Code\\Data\\data.json"
file_path_hero = "C:\\Users\\dcrai\\source\\repos\\DATA698\\Code\\Data\\hero_id_table.csv"
#file_csv = "C:\\Users\\dcrai\\source\\repos\\DATA698\\Code\\Data\\iter_1.csv"
#data = json.loads(file)

with open(file_path, 'r') as file:
    data = json.load(file)


heroes= pd.read_csv(file_path_hero)
# Now 'data' contains the contents of the JSON file


# %%
# Extract 'match_id', 'hero_id', and 'gold_t' from each element in 'data'
match_ids = [element['match_id'] for element in data]
hero_ids = [element['hero_id'] for element in data]
gold_t_values = [element['gold_t'] for element in data]

# Create a DataFrame from the extracted values
df = pd.DataFrame({'match_id': match_ids, 'hero_id': hero_ids, 'gold_t': gold_t_values})

# Displ

#7517376613

# %%
df

# %%
#df_match = df[df['match_id'] == 7517376613]

df_match = df[:300]

#df_match

df_subset = df_match[['hero_id', 'gold_t']].copy()

df_t = df_subset.T

#df_t.columns
# explode = df_t.explode(26)

# explode
df_subset

# %% [markdown]
# ### TimeSeriesDataset Building

# %% [markdown]
# Summary of the classes we are about to build.
# 
# General Things to Understand:
# 
# - the forward method is basically a return call
#     - forward method is a part of all nn.Module classes (nn.Module is the base class for any neural net node)
# 
# - 
# 
# 1. TimeSeriesDataset
# 
# The TimeSeriesDataset is a class we are making while inheriting the PyTorch Dataset class, which is an abstract free-form class with little restriction to it that mainly provides certain methods like __get_item__ that can be passed on to subsequent layers in the architecture. 
# 
# Class Responsibiltiies:
# - Establish hero_ids and time_series as separate callable variables
# - Establish the length of the Longest tensor to help with padding
# - Establish retrieving a hero_id and its tensor with __get_item__
# 
# When get_item is called by DataLoader it will be passed an index, where it will retrieve a hero_id and its time series. Then it pads the time series based on the length of the longest tensor and returns the hero_id, padded time series, and the time series' length before padding
# 
# 2. Process Embedding
# 
# The ProcessEmbedding is a class also provided by PyTorch that will act as an embedding handler. The embedding matrix is a matrix of vectors containing numerical values that can be trained over to help capture the unique effects of each hero. Effectively, this is the method used to account for heros that are more eFfective with gold than others.
# 
# Class Responsibilities:
# - Establish number of hero_ids/vectors needed in the matrix
# - Establish number of dimensions for the embedding matrix to have
# - Instantiate the nn.Embeddings class with the prior dimensions

# %%
from torch.nn.utils.rnn import pack_padded_sequence
pad_test = [torch.tensor(ts) for ts in df_subset['gold_t']]

#pad_test

# %%
tensor_pad_test = pad_test[1]

# %%
pad_max_length = max(len(ts) for ts in pad_test)
print(pad_max_length)
padded_time_series = torch.zeros(pad_max_length)
ts_length = len(tensor_pad_test)
padded_time_series[:ts_length] = tensor_pad_test

#padded_time_series

#pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True)


# %%
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    # Class to create our dataset
    def __init__(self, df):
        self.hero_ids = df['hero_id'].values # Declaring hero_id values
        self.time_series = [torch.tensor(ts) for ts in df['gold_t']] # Converting the time_series into Tensors
        self.max_length = max(len(ts) for ts in self.time_series) # Grabs max length of all the tensors to pad them with 0s later

        """
        # Check the dimensionality of the time series tensors
        for ts in self.time_series:
            print(f"Time series shape: {ts.shape}, Dimensions: {ts.dim()}")
        """


    def __len__(self):
        return len(self.hero_ids) # Convenient length call
    

    def __getitem__(self, idx):
        # For the DataLoader, indexes are called to pull time-series data. Since we have the categorical variable in hero_id, we need to ensure
        #     to pull both the hero_id and the associated index of the time series. To do that the DataLoader passes the __getitem__ method an
        #     index, and __getitem__ returns the hero_id and the time_series at that index. In a time-series without categorical data, it'd normally
        #     just be the time series that is returned.
        hero_id = self.hero_ids[idx]
        time_series = self.time_series[idx]
        length = len(time_series)

        padded_time_series = torch.zeros(self.max_length)
        padded_time_series[:length] = time_series

        return hero_id, padded_time_series, length

# %% [markdown]
# Playing with how the __getitem__ method works below

# %%
# ts_test = [torch.tensor(ts) for ts in df_subset['gold_t']] # Converting the time_series into Tensors
# length = len(ts_test[0])
# max_length = max(len(ts) for ts in ts_test)
# print(ts_test[0].shape)
# #print(ts_test[0].size(1))

# padded_test = torch.zeros(max_length, ts_test[0].size(0)) 

# padded_test[:length] = ts_test[0]

# padded_test
# #padded_test.size(1)
# # ts_test[1].size(1)
# # max_length = max(len(ts) for ts in ts_test)

# # print(max_length)
# # print(ts_test[1].size(1))

# # len(ts_test)

# %% [markdown]
# ### Embedding Processing
# 
# Embedding will allow the model to develop a representation for each hero_id, with like hero_ids having similar embedding vectors. These vectors will help represent the differences between carries, mids, offlaners, and more. The embedding matrix is treated as a learning parameter, which means backpropagation will update the values in the matrix based on the loss function in training.
# 
# Rule of Thumb: A common rule of thumb is to set the embedding_dim to be at least as large as the logarithm of the number of classes or groups you want to represent or expect. In your case, with around 5 groups, the logarithm of 5 is approximately 2. Since I know there are 5 roles, and I'd like to see them expressed atleast by role the log would be an embed vector of size 2.
# 
# Rule of Thumb (NLP): A common rule of thumb in natural language processing (NLP) tasks, where embeddings are widely used, is to set the embedding dimension to be approximately the fourth root of the vocabulary size (number of unique tokens/entities). While this is not a hard rule, it provides a reasonable ballpark estimate. In your case, with 124 unique processes, the fourth root of 124 is approximately 5. Multiplying this by a factor (e.g., 16 or 32, which are common embedding sizes) gives you a range of 80-160 as a reasonable starting point for the embedding dimension.
# 
# In many deep learning applications, embedding dimensions of 64, 128, or 256 are used.
# 
# I will try embed sizes of 20, 32, 64, 128, and 256.

# %%
print(list(enumerate(df_subset['hero_id'].unique())))

#(0, 26), (1, 84), (2, 25), (3, 8

matrix_test = {hero_id: idx for idx, hero_id in enumerate(df_subset['hero_id'].unique())}
print(matrix_test)
               

# %%


matrix_test = {hero_id: idx for idx, hero_id in enumerate(df_subset['hero_id'].unique())}

# %%
import torch
import torch.nn as nn

class ProcessEmbedding(nn.Module):
    def __init__(self, df, embedding_dim):
        super(ProcessEmbedding, self).__init__() 
        """
        super() calls the intialization of the parent class of ProcessEmbedding, in this case nn.Module (PyTorch's class for all nnets)
        This is done to initizialize the class correctly. If not called, the nn.Module's functionalities will not work.

        """
        self.num_processes = len(df['hero_id'].unique()) # declaring number of different categories of time-series for dimensionialty reasons
        self.embedding_dim = embedding_dim # passing our embed size to be a class attribute
        self.process_embeddings = nn.Embedding(self.num_processes, embedding_dim)

        """ nn.Embedding creates and stores the embedding vector , it takes two arguments;
                    num_processes - the number of different embeddings it will need to hold (for us it would be the number of hero_ids = 124)
                    embedding_dim - the size of the vector of the embedding
        """


        self.hero_id_to_idx = {hero_id: idx for idx, hero_id in enumerate(df['hero_id'].unique())}

        """  self.hero_id_to_idx: a mapping of hero_ids to embedding matrix vector IDs
        - converts hero_id to an integer index and creates a matrix mapping of the values to pass on
        - is created since nn.Embedding expects an integer for an index value where an embedding is stored
        - this attribute effectively converts hero_ids to an index value that corresponds to the Embedding vectors row of values for that hero_id

        enumerate(df['hero_id'].unique()): iterates over the unique hero_id values and assigns a sequential index (idx) to each value, starting from 0. 
        {} creates a dictionary of key-pairs, the basic form of creating a dictionary: {key_expression: value_expression for item in iterable},
            which allows us to define the key-pairs over an iterable in one line of code


        It returns a list of tuples (idx, hero_id).
        For example, if the unique hero_id values are ['026', '184', '225', '38'], the enumerate function will return:
        {26: 0, 184: 1, 225: 2, 38: 3} - thus creating our mapping to pull the correct embedding vector per the hero id
        
        """
        

    def forward(self, hero_ids):
        """
        Called when: 
            LSTM Module calls process_embedding
        Args:
            hero_ids (Tensor or List): A tensor or list of hero IDs.
        Returns:
            Tensor: A tensor of shape (batch_size, embedding_dim) containing the process embeddings.

        We use the embedding with the forward method below.
            1. We convert the input hero_ids to their corresponding indices (process_ids) in the Embed Vector using the self.hero_id_to_idx mapping.
            2. We create a PyTorch tensor process_ids from the indices.
            3. We pass this process_ids tensor to the self.process_embeddings module, which retrieves and returns the corresponding embedding vectors.
        """
        process_ids = []
        print(hero_ids)
        for hero_id in hero_ids:
            process_id = self.hero_id_to_idx[hero_id.item()] # pull the associated index number to the hero_id that will later be used as the index identifier in the embedding matrix
            if process_id not in process_ids: # this will ensure that only unique values are added to the list
                process_ids.append(process_id)
        
        # remove or no? - process_ids = [self.hero_id_to_idx[hero_id] for hero_id in hero_ids] # convert hero_ids - remove or no?
        process_ids = torch.tensor(process_ids) # Create tensor of process_ids
        return self.process_embeddings(process_ids) # Pass the process_ids to self.process_embeddings
    

    # def forward(self, hero_ids):
    #     process_ids = []
    #     for hero_id in hero_ids:
    #         try:
    #             process_id = self.hero_id_to_idx[hero_id.item()]
    #             process_ids.append(process_id)
    #         except KeyError:
    #             print(f"Error: hero_id {hero_id.item()} not found in hero_id_to_idx dictionary.")
    #             raise

    #     process_ids = torch.tensor(process_ids)
    #     return self.process_embeddings(process_ids)

# %% [markdown]
# #### Data Loader

# %%
dataset = TimeSeriesDataset(df_subset)
process_embedding = ProcessEmbedding(df_subset, embedding_dim=84)

# %% [markdown]
# #### Processing The Embedding

# %%
batch_size = 4
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True) 

for hero_ids, time_series, lengths in data_loader:
    process_embeddings = process_embedding(hero_ids)
    # Use process_embeddings and time_series for training or inference


# %% [markdown]
# ## Building LSTM Model

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
        
        # Reshape process embeddings to match the input shape of LSTM
        process_embeddings = process_embeddings.unsqueeze(1).repeat(1, seq_length, 1)
        
        # Concatenate process embeddings with time series data
        input_data = torch.cat((process_embeddings, time_series), dim=-1)
        
        # Pack the padded sequences
        packed_input = pack_padded_sequence(input_data, lengths, batch_first=True, enforce_sorted=False)


        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        # 
        packed_output, _ = self.lstm(packed_input, (h0, c0))

        # Unpack the output
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Take the last output of the LSTM
        out = self.fc(out[:, -1, :])
        
        return out

# %%
time_series.shape

# %%
input_size = process_embedding.embedding_dim + time_series.shape[-1]  # Number of features (embedding_dim + time_series_dim)
hidden_size = 64
num_layers = 2
output_size = 1  # Assuming you want to predict a single value

model = LSTMModel(input_size, hidden_size, num_layers, output_size, process_embedding)

# %% [markdown]
# #### Training the Model

# %% [markdown]
# ### Training Parameters

# %%
# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
batch_size = 32

# %% [markdown]
# ### Padded Training Loop

# %% [markdown]
# 

# %% [markdown]
# #### Evaluating Model

# %% [markdown]
# 

# %% [markdown]
# ## Running LSTM

# %%
test_size = .30


# %% [markdown]
# ### Train Test Split

# %%
train_df, test_df = train_test_split(df_subset, test_size=test_size, shuffle=False)

# %% [markdown]
# #### Dataset and Data Load

# %%
train_dataset = TimeSeriesDataset(train_df)
test_dataset = TimeSeriesDataset(test_df)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


