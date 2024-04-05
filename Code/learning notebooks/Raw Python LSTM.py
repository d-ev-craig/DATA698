
## Libraries
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

## Load Raw Data
file_path = "C:\\Users\\dcrai\\source\\repos\\DATA698\\Code\\Data\\data.json"
file_path_hero = "C:\\Users\\dcrai\\source\\repos\\DATA698\\Code\\Data\\hero_id_table.csv"
#file_csv = "C:\\Users\\dcrai\\source\\repos\\DATA698\\Code\\Data\\iter_1.csv"
#data = json.loads(file)

with open(file_path, 'r') as file:
    data = json.load(file)


heroes= pd.read_csv(file_path_hero)
# Now 'data' contains the contents of the JSON file


## Clean up Data

# Extract 'match_id', 'hero_id', and 'gold_t' from each element in 'data'
match_ids = [element['match_id'] for element in data]
hero_ids = [element['hero_id'] for element in data]
gold_t_values = [element['gold_t'] for element in data]

# Create a DataFrame from the extracted values
df = pd.DataFrame({'match_id': match_ids, 'hero_id': hero_ids, 'gold_t': gold_t_values})

#7517376613

## Subset Data for Testing

#df_match = df[df['match_id'] == 7517376613]

df_match = df[:300]

#df_match

df_subset = df_match[['hero_id', 'gold_t']].copy()

df_t = df_subset.T

#df_t.columns
# explode = df_t.explode(26)

# explode
df_subset



## Defining TimeSEries Dataset

import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    # Class to create our dataset
    def __init__(self, df):
        self.hero_ids = df['hero_id'].values # Declaring hero_id values
        self.time_series = [torch.tensor(ts) for ts in df['gold_t']] # Converting the time_series into Tensors
        self.max_length = max(len(ts) for ts in self.time_series) # Grabs max length of all the tensors to pad them with 0s later


    def __len__(self):
        return len(self.hero_ids) # Convenient length call

    def __getitem__(self, idx): # getitem is called by the PyTorch Dataloader
        # For the DataLoader, indexes are called to pull time-series data. Since we have the categorical variable in hero_id, we need to ensure
        #     to pull both the hero_id and the associated index of the time series. To do that the DataLoader passes the __getitem__ method an
        #     index, and __getitem__ returns the hero_id and the time_series at that index. In a time-series without categorical data, it'd normally
        #     just be the time series that is returned.
        hero_id = self.hero_ids[idx]
        time_series = self.time_series[idx]
        length = len(time_series)

        # Below is where padding is handled.
        padded_time_series = torch.zeros(self.max_length, time_series.size(0)) # Creates a time-series of zeros of the longest series in the set
        padded_time_series[:length] = time_series #assigns the values from the individual time_series to the first 
        return hero_id, padded_time_series, length

## Defining Process Embedding Class

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

        """  self.hero_id_to_idx; converts hero_id to an integer index
        - is created since nn.Embedding expects an integer for an index value where an embedding is stored
        - this attribute effectively converts hero_ids to an index value that corresponds to the Embedding vectors row of values for that hero_id

        enumerate(df['hero_id'].unique()) iterates over the unique hero_id values and assigns a sequential index (idx) to each value, starting from 0. 
        It returns a list of tuples (idx, hero_id).
        For example, if the unique hero_id values are ['026', '184', '225', '38'], the enumerate function will return:
        [(0, '026'), (1, '184'), (2, '225'), (3, '38')]
        
        """
        

    def forward(self, hero_ids):
        """
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
            process_id = self.hero_id_to_idx[hero_id.item()]
            process_ids.append(process_id)
        # process_ids = [self.hero_id_to_idx[hero_id] for hero_id in hero_ids] # convert hero_ids
        process_ids = torch.tensor(process_ids.unique) # Create tensor of process_ids
        return self.process_embeddings(process_ids) # Pass the process_ids to self.process_embeddings
    


## Instantiate dataset & process embedding

dataset = TimeSeriesDataset(df_subset)
process_embedding = ProcessEmbedding(df_subset, embedding_dim=84)


## Processing Embeddings

batch_size = 4
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True) 

for hero_ids, time_series in data_loader:
    process_embeddings = process_embedding(hero_ids)
    # Use process_embeddings and time_series for training or inference