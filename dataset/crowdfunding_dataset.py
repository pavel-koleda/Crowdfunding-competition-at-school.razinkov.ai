import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering

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

        

        # df = df.sample(3000)
        if set_type is not SetType.test:
            date_threshold = pd.to_datetime('2021-01-01')
            df = df[df['launched_at'] > date_threshold]
            df = df[["creator_id", "fx_rate", "static_usd_rate", "goal", "name", "blurb", "created_at", "launched_at", "deadline", "country", "state"]]
            df['state'] = df['state'].map({'successful': 1, 'failed': 0})
        else:
            ids = df['id'].tolist()
            df = df[["creator_id", "fx_rate", "static_usd_rate", "goal", "name", "blurb", "created_at", "launched_at", "deadline", "country"]]
            
        df['blurb'] = df['name'] + " " + df['blurb']

        # 2. Drop duplicates from dataframe
        df = df.drop_duplicates()
        

        creator_counts = df['creator_id'].value_counts()
        df['user_frequency'] = df['creator_id'].map(creator_counts)


        if set_type is SetType.train:
            self.countries = df['country'].unique().tolist()
            df['blurb'] = df['blurb'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
            
            vectorizer = TfidfVectorizer(
                # stop_words='english',  # Удаление стоп-слов
                analyzer='char_wb',
                ngram_range=(2, 5),
                max_features=100,       # Ограничение на количество признаков
                max_df=0.8,            # Исключение слишком частых слов
                min_df=0.03             # Исключение слишком редких слов
            )
            
            tfidf_matrix = vectorizer.fit_transform(df['blurb'])
            
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
            
            df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
            self.vectorizer = vectorizer
        else:
            tfidf_matrix = self.vectorizer.transform(df['blurb'])
            
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=self.vectorizer.get_feature_names_out())
            
            df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)


        df = df.drop(columns=['name', 'blurb'])
        # 3. Convert categorical feature columns 
        df['country'] = pd.Categorical(df['country'], categories=self.countries, ordered=False)

        # 4. Convert categorical features to one-hot encoding vectors 

        df = pd.get_dummies(df, columns=['country'])


        # 5. Create features 


        df['project_duration'] = (df['deadline'] - df['launched_at']).dt.days
        df['project_atart_duration'] = (df['launched_at'] - df['created_at']).dt.days


        df['launched_at_sin'] = np.sin(2 * np.pi * df['launched_at'].dt.dayofyear / 365)
        df['launched_at_cos'] = np.cos(2 * np.pi * df['launched_at'].dt.dayofyear / 365)
        df['deadline_sin'] = np.sin(2 * np.pi * df['deadline'].dt.dayofyear / 365)
        df['deadline_cos'] = np.cos(2 * np.pi * df['deadline'].dt.dayofyear / 365)


        print(set_type)

        if set_type is SetType.train:
            features = self.preprocessing.train(df.drop(columns=['state']).to_numpy(dtype=np.float32))
        elif set_type is SetType.validation:
            features = self.preprocessing(df.drop(columns=['state']).to_numpy(dtype=np.float32))
        else:
            features = self.preprocessing(df.to_numpy(dtype=np.float32)), ids
        
        if set_type is not SetType.test:
            targets = df['state'].to_numpy(dtype=int)
        else:
            targets = np.array([])

        return {'features': features, 'targets': targets}

    def __call__(self, set_type: str) -> dict:
        """Returns preprocessed data."""
        return self.data[set_type]
