import os
import pandas as pd


data_dir = "data/yellow_tripdata/"

# choosing a subset for now
files_to_load = [
    "yellow_tripdata_2019-01.csv",
    "yellow_tripdata_2019-02.csv",
    "yellow_tripdata_2019-03.csv",
    "yellow_tripdata_2019-04.csv",
    "yellow_tripdata_2019-05.csv",
    
    
]


dfs = []
for file_name in files_to_load:
    file_path = os.path.join(data_dir, file_name)
    df = pd.read_csv(file_path)
    dfs.append(df)


taxi_data = pd.concat(dfs, ignore_index=True)


columns_to_drop = ["VendorID", "RatecodeID", "store_and_fwd_flag", "payment_type",
                   "fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount",
                   "improvement_surcharge", "total_amount", "congestion_surcharge"]
#columns_to_drop = [col for col in columns_to_drop if col in taxi_data.columns]  # Check if column exists
taxi_data.drop(columns=columns_to_drop, inplace=True)



taxi_data["tpep_pickup_datetime"] = pd.to_datetime(taxi_data["tpep_pickup_datetime"])
taxi_data["tpep_dropoff_datetime"] = pd.to_datetime(taxi_data["tpep_dropoff_datetime"])

# Filter out records with invalid or missing data
taxi_data = taxi_data.dropna()

# Filter out trips with unrealistic distances
taxi_data = taxi_data[(taxi_data["trip_distance"] > 0)]


print("Preprocessed Taxi Trip Data:")
print(taxi_data.info())
print(taxi_data.head())
