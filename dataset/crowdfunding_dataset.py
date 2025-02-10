import os
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering

from utils.common_functions import read_dataframe_file
from utils.enums import SetType
from utils.preprocessing import Preprocessing


class AuthorsProjectsHistory:

    def __init__(self):
        self.history = defaultdict(list)

    def fit(self, df):
        # Логика с iterrows работает ооочень медленно, но вроде терпимо
        for index, row in df.sort_values(by=['deadline']).reset_index(drop=True).iterrows():
            author_id = row['creator_id']
            result = row['state']
            date = row['deadline']

            self.history[author_id].append((date, result))

    def transform(self, df):
        suc_results = []
        fail_results = []
        for index, row in df.sort_values(by=['deadline']).reset_index(drop=True).iterrows():
            author_id = row['creator_id']
            deadline_ts = row['deadline']  # TODO: мб тут брать старт?
            author_history = self.history[author_id]

            succeed, failed = 0, 0
            for date, result in author_history:
                if deadline_ts <= date:
                    break
                else:
                    succeed += int(result == 1)
                    failed += int(result == 0)
            suc_results.append(succeed)
            fail_results.append(failed)
        return np.array(fail_results), np.array(suc_results)


def date_to_bucket(date, min_year, window_size):
    return (date.month + (date.year - min_year) * 12) // window_size


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
        if set_type is not SetType.test:
            df['state'] = df['state'].map({'successful': 1, 'failed': 0})
        else:
            ids = df['id'].tolist()

        # Распаковка json
        c = pd.json_normalize(df['category'])
        c.columns = [f'category_{x}' for x in c.columns]
        df = pd.concat([df, c], axis=1).drop(columns=['category'])

        l = pd.json_normalize(df['location'])
        l.columns = [f'location_{x}' for x in l.columns]
        df = pd.concat([df, l], axis=1).drop(columns=['location'])

        # Удаление пропусков
        df['location_type'] = df['location_type'].fillna('Unknown')
        df['category_parent_id'] = df['category_parent_id'].fillna(9999)

        # Признаки времени
        df['duration_ld'] = (df['deadline'] - df['launched_at']).dt.total_seconds()
        df['duration_cl'] = (df['launched_at'] - df['created_at']).dt.total_seconds()

        df['duration_cl_log'] = np.log(df['duration_cl'] + 1)
        df['duration_cl'] = np.clip(df['duration_cl'], 0, 60 * 60 * 24 * 100)

        df['launched_at_sin'] = np.sin(2 * np.pi * df['launched_at'].dt.dayofyear / 365)
        df['launched_at_cos'] = np.cos(2 * np.pi * df['launched_at'].dt.dayofyear / 365)
        df['deadline_sin'] = np.sin(2 * np.pi * df['deadline'].dt.dayofyear / 365)
        df['deadline_cos'] = np.cos(2 * np.pi * df['deadline'].dt.dayofyear / 365)

        # Валюта
        df['val1'] = df['goal'] / df['fx_rate']
        df['val2'] = df['goal'] / df['static_usd_rate']
        df['val3'] = df['goal'] * df['fx_rate']
        df['val4'] = df['goal'] * df['fx_rate']

        df['bucket'] = df['deadline'].apply(lambda x: date_to_bucket(x, 2009, 3))
        tmp = df[['category_parent_id', 'bucket', 'val1', 'val2', 'val3', 'val4']].copy()

        if set_type == SetType.train:
            self.agg = tmp.groupby(['category_parent_id', 'bucket']).agg({
                'val1': 'mean',
                'val2': 'mean',
                'val3': 'mean',
                'val4': 'mean',
            }).fillna(1.0).reset_index()
            self.agg = self.agg.rename(columns={
                'val1': 'val1_mean',
                'val2': 'val2_mean',
                'val3': 'val3_mean',
                'val4': 'val4_mean',
            })

        df = pd.merge(df, self.agg, on=['category_parent_id', 'bucket'], how='left')
        df[['val1_mean', 'val2_mean', 'val3_mean', 'val4_mean']] = df[['val1_mean', 'val2_mean', 'val3_mean', 'val4_mean']].fillna(1)
        df['val1/val1_mean'] = df['val1'] / df['val1_mean']
        df['val2/val2_mean'] = df['val2'] / df['val2_mean']
        df['val3/val3_mean'] = df['val3'] / df['val3_mean']
        df['val4/val4_mean'] = df['val4'] / df['val4_mean']

        # Авторы
        creator_counts = df['creator_id'].value_counts()
        df['user_frequency'] = df['creator_id'].map(creator_counts)

        if set_type is SetType.train:
            self.authors_project_history = AuthorsProjectsHistory()
            self.authors_project_history.fit(df)
        failed_results, succeed_results = self.authors_project_history.transform(df)
        df['author_succeed_results'] = succeed_results
        df['author_failed_results'] = failed_results
        df['author_succeed_ratio'] = (df['author_succeed_results'] + 1) / (
                df['author_failed_results'] + df['author_succeed_results'] + 2)
        df['author_ratio'] = (df['author_succeed_results'] + 1) / (df['author_failed_results'] + 1)

        # Тексты
        df['blurb'] = df['name'] + " " + df['blurb']
        if set_type is SetType.train:
            df['blurb'] = df['blurb'].str.lower().str.replace(r'[^\w\s]', '', regex=True)

            vectorizer = TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(2, 5),
                max_features=200,
                max_df=0.98,
                min_df=0.015,
            )

            tfidf_matrix = vectorizer.fit_transform(df['blurb'])

            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

            df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
            self.vectorizer = vectorizer
        else:
            tfidf_matrix = self.vectorizer.transform(df['blurb'])

            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=self.vectorizer.get_feature_names_out())

            df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

        df_success = read_dataframe_file(r'dataset\kikstarter_Success_stats.csv')
        df_dollars = read_dataframe_file(r'dataset\kikstarter_Dollars_stats.csv')
        df = pd.merge(df, df_success, left_on='category_parent_name', right_on='Category', how='left')
        df = df.drop(columns='Category')
        df = pd.merge(df, df_dollars, left_on='category_parent_name', right_on='Category', how='left')
        df = df.drop(columns='Category')

        if set_type is SetType.train:
            self.fill_na_dict = dict()
            for cat in df_success.columns.drop(['Category']):
                self.fill_na_dict[cat] = df[cat].mean()

            for cat in df_dollars.columns.drop(['Category']):
                self.fill_na_dict[cat] = df[cat].mean()

        for k, v in self.fill_na_dict.items():
            df[k] = df[k].fillna(v)

        # Категориальные фичи
        if set_type == SetType.train:
            self.country = df['country'].unique().tolist()
            self.currency = df['currency'].unique().tolist()
            self.category_id = df['category_id'].unique().tolist()
            self.category_parent_id = df['category_parent_id'].unique().tolist()
            self.month_start = df['created_at'].dt.month.unique().tolist()
            self.year_start = df['created_at'].dt.year.unique().tolist()
            self.location_type = df['location_type'].unique().tolist()

        df['country'] = pd.Categorical(df['country'], categories=self.country, ordered=False)
        df['currency'] = pd.Categorical(df['currency'], categories=self.currency, ordered=False)
        df['category_id'] = pd.Categorical(df['category_id'], categories=self.category_id, ordered=False)
        df['category_parent_id'] = pd.Categorical(df['category_parent_id'], categories=self.category_parent_id,
                                                  ordered=False)
        df['month_start'] = pd.Categorical(df['created_at'].dt.month, categories=self.month_start, ordered=False)
        df['year_start'] = pd.Categorical(df['created_at'].dt.year, categories=self.year_start, ordered=True)
        df['location_type'] = pd.Categorical(df['location_type'], categories=self.location_type, ordered=False)

        if set_type is SetType.train:
            df = df[df['year_start'].isin(list(range(2022, 2024 + 1)))]
        # TODO: Убрать static_usd и fx_rate?
        df = df.drop(
            columns=['id', 'blurb', 'created_at', 'currency_symbol', 'deadline', 'launched_at', 'name', 'photo', 'slug',
                     'discover_category_url', 'urls', 'creator', 'creator_id', 'creator_name', 'category_alt_parent_id',
                     'category_name', 'category_parent_name', 'category_slug', 'category_urls.web.discover',
                     'location_country', 'location_displayable_name', 'location_expanded_country', 'location_id',
                     'location_is_root', 'location_name', 'location_short_name', 'location_slug', 'location_state',
                     'location_urls.api.nearby_projects', 'location_urls.web.discover', 'location_urls.web.location',
                     'bucket', 'disable_communication', 'is_starrable'
                     ])
        df = pd.get_dummies(df)

        # TODO: Не делать препроцессинг для OHE фич?
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
