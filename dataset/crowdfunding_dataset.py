import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.common_functions import read_dataframe_file
from utils.enums import SetType
from utils.preprocessing import Preprocessing


class CrowdfundingDataset:
    """A class for the Crowdfunding dataset. This class reads the data and preprocesses it."""

    def __init__(self, config):
        """Initializes the Crowdfunding dataset class instance."""
        self.config = config
        
        # Preprocessing class initialization
        self.preprocessing = Preprocessing(config.preprocess_type)
        
        # Reads the data
        self.data = {}
        for set_type in SetType:
            self.data[set_type.name] = self.dataframe_preprocessing(
                os.path.join(config.path_to_data, config.type[set_type.name]), set_type
            )


    def dataframe_preprocessing(self, path_to_dataframe: str, set_type: SetType) -> dict:
        """Reads and preprocesses the data.

        Args:
            set_type: Data set_type from SetType.

        Returns:
            A dict with the following data:
                {'features': images (numpy.ndarray), 'targets': targets (numpy.ndarray), 'paths': list of paths}
        """
        # 1. Read a dataframe file
        df = read_dataframe_file(path_to_dataframe)

        

        # df = df[:10000]
        if set_type is not SetType.test:
            date_threshold = pd.to_datetime('2021-01-01')
            df = df[df['launched_at'] > date_threshold]
            df = df[["created_at", "launched_at", "deadline", "country", "state"]]
            df['state'] = df['state'].map({'successful': 1, 'failed': 0})
        else:
            ids = df['id'].tolist()
            df = df[["created_at", "launched_at", "deadline", "country"]]
        # 2.1 Drop outliers
        # df = df.drop(columns=['store_and_fwd_flag', 'improvement_surcharge', 'congestion_surcharge'])
        # 2. Drop duplicates from dataframe
        df = df.drop_duplicates()
        
        # if set_type is not SetType.test:

            # df = df[df['trip_distance'] > 0]
            # df = df[df['amount_to_pay'] > 0]
            # df = df[df['fare_amount'] > 0]
            # # df = df[df['improvement_surcharge'] > 0]
            # # df = df[df['congestion_surcharge'] > 0]
            # df = df[df['tpep_pickup_datetime'] < df['tpep_dropoff_datetime']]

            # # Уберем тех кто дал чаевых больше 50%, таких только 1000
            # df = df[~((df['amount_to_pay'] * 0.5 < df['tip_amount']) & (df['tip_amount'] > 25))]
        # if 'price' in df.columns:
        if set_type is SetType.train:
            self.countries = df['country'].unique().tolist()

        # 3. Convert categorical feature columns (['color', 'clarity', 'cut']) into pd.Categorical.
        df['country'] = pd.Categorical(df['country'], categories=self.countries, ordered=False)
        # df['pickup_day_of_week'] = pd.Categorical(df['tpep_pickup_datetime'].dt.dayofweek, categories=self.config.categories['pickup_day_of_week'], ordered=False)
        # # 4. Convert categorical features to one-hot encoding vectors (columns ['color', 'clarity', 'cut'])
        df = pd.get_dummies(df)

        # 5. Create features from all columns except for the 'price'
        # # Получаем минимальный и максимальный год из данных
        # min_year = df['tpep_pickup_datetime'].dt.year.min()
        # max_year = df['tpep_pickup_datetime'].dt.year.max()

        # # Создаем множество праздничных дат для всех необходимых лет
        # holiday_dates = set()
        # for year in range(min_year, max_year + 1):
        #     holiday_dates.update(holidays.US(years=year).keys())

        # Векторизированная операция проверки
        # df['is_holiday'] = df['tpep_pickup_datetime'].dt.date.isin(holiday_dates).astype(int)

        # Calculate trip duration in minutes
        df['project_duration'] = (df['deadline'] - df['launched_at']).dt.days
        df['project_atart_duration'] = (df['launched_at'] - df['created_at']).dt.days

        # Extract day of week and hour from pickup time
        # df['pickup_day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek
        # df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
        # Преобразование часов в синусоиды
        df['launched_at_sin'] = np.sin(2 * np.pi * df['launched_at'].dt.dayofyear / 365)
        df['launched_at_cos'] = np.cos(2 * np.pi * df['launched_at'].dt.dayofyear / 365)

        # df['is_night'] = (df['tpep_pickup_datetime'].dt.hour >= 22) | (df['tpep_pickup_datetime'].dt.hour < 6)
        # # Calculate speed in mph
        # df['speed_mph'] = df['trip_distance'] / (df['trip_duration_in_minutes'] / 60)
        # if set_type is not SetType.test:
        #     df = df[df['speed_mph'] < 60]
        #     df = df[~((df['trip_duration_in_minutes'] > 200) & (df['speed_mph'] < 3))]
            
        # df['trip_distance'] = np.ceil(df['trip_distance'] / 1)
        
        # df = df.drop(columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])
        # df = df.drop(columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'PULocationID', 'DOLocationID'])
        # Calculate the average tip amount for each (PULocationID, DOLocationID) combination
        # if set_type is SetType.train:
        #     # print(1)
        #     average_tips = df.groupby(['PULocationID', 'DOLocationID'])['tip_amount'].mean().reset_index()
        #     average_tips.rename(columns={'tip_amount': 'average_tip'}, inplace=True)

        #     # Add the average tips back to the original dataset
        #     df = df.merge(average_tips, on=['PULocationID', 'DOLocationID'], how='left')

        #     # Save the average tips for inference
        #     average_tips_file = r'data\average_tips.csv'
        #     average_tips.to_csv(average_tips_file, index=False)
        
        # if set_type is SetType.train:
        #     self.PULocationID_mean = df.groupby('PULocationID')['tip_amount'].mean()
        #     self.DOLocationID_mean = df.groupby('DOLocationID')['tip_amount'].mean()

        #     df['PULocationID_encoded'] = df['PULocationID'].map(self.PULocationID_mean)
        #     df['DOLocationID_encoded'] = df['DOLocationID'].map(self.DOLocationID_mean)
        # else:
        #     df['PULocationID_encoded'] = df['PULocationID'].map(self.PULocationID_mean).fillna(self.PULocationID_mean.mean())
        #     df['DOLocationID_encoded'] = df['DOLocationID'].map(self.DOLocationID_mean).fillna(self.DOLocationID_mean.mean())



        # average_tips = pd.read_csv(r'data\average_tips.csv')
        # df = df.drop(columns=['PULocationID', 'DOLocationID'])
        # features = df.drop(columns=['price']).to_numpy(dtype=float)
        print(set_type)
        # print(df.drop(columns=['state']).to_numpy(dtype=np.float32).head(2))
        # print(set_type)
        # print(df.index.tolist()[:5])
        if set_type is SetType.train:
            features = self.preprocessing.train(df.drop(columns=['state']).to_numpy(dtype=np.float32))
        elif set_type is SetType.validation:
            # origin_index = df.index
            # df = df.merge(average_tips, on=['PULocationID', 'DOLocationID'], how='left')
            # df.index = origin_index
            # df['average_tip'] = df['average_tip'].fillna(average_tips['average_tip'].mean())
            features = self.preprocessing(df.drop(columns=['state']).to_numpy(dtype=np.float32))
        else:
            # origin_index = df.index
            # df = df.merge(average_tips, on=['PULocationID', 'DOLocationID'], how='left')
            # df.index = origin_index
            # df['average_tip'] = df['average_tip'].fillna(average_tips['average_tip'].mean())
            # print(df.index.tolist()[:5])
            features = self.preprocessing(df.to_numpy(dtype=np.float32)), ids
        
        if set_type is not SetType.test:
            targets = df['state'].to_numpy(dtype=int)
            # print(targets)
        else:
            targets = np.array([])

        return {'features': features, 'targets': targets}

    def __call__(self, set_type: str) -> dict:
        """Returns preprocessed data."""
        return self.data[set_type]
