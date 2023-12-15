import pandas
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import cv2
import os
from nltk import wordpunct_tokenize
import re

class MLDataset(Dataset):

    def __init__(self, is_train=True):
        users = pandas.read_csv('dataset/users.dat', sep='::',
                        engine='python',
                        names=['userid', 'gender', 'age', 'occupation', 'zip']).set_index('userid')
        ratings = pandas.read_csv('dataset/ratings.dat', engine='python',
                                sep='::', names=['userid', 'movieid', 'rating', 'timestamp'])
        self.movies_train = pandas.read_csv('dataset/movies_train.dat', engine='python',
                                sep='::', names=['movieid', 'title', 'genre'], encoding='latin-1', index_col=False).set_index('movieid')
        self.movies_test = pandas.read_csv('dataset/movies_test.dat', engine='python',
                                sep='::', names=['movieid', 'title', 'genre'], encoding='latin-1', index_col=False).set_index('movieid')
        self.movies_train['genre'] = self.movies_train.genre.str.split('|')
        self.movies_test['genre'] = self.movies_test.genre.str.split('|')

        users.age = users.age.astype('category')
        users.gender = users.gender.astype('category')
        users.occupation = users.occupation.astype('category')
        ratings.movieid = ratings.movieid.astype('category')
        ratings.userid = ratings.userid.astype('category')

        folder_img_path = 'dataset/ml1m-images'
        self.movies_test['id'] = self.movies_test.index
        self.movies_test.reset_index(inplace=True)
        self.movies_test['img_path'] = self.movies_test.apply(lambda row: os.path.join(folder_img_path, f'{row.id}.jpg'), axis = 1)

        if is_train:
            self.data =  self.movies_train
        else:
            self.data = self.movies_test
        self.data['title_tokens'] = [self.tokenize(x) for x in self.data.title]

        # create vocab
        vocab = self.create_vocab()
        pad_token = '<PAD>'
        unk_token = '<UNK>'
        self.token2idx = {token: idx for idx, token in enumerate(vocab)}

        # Create a binary vector for each word in each sentence
        MAX_LENGTH = 7
        vectors = []
        for title_tokens in self.data.title_tokens.tolist():
            if len(title_tokens) < MAX_LENGTH:
                num_pad = MAX_LENGTH - len(title_tokens)
                for idx in range(num_pad):
                    title_tokens.append(pad_token)
            else:
                title_tokens = title_tokens[:MAX_LENGTH]
            title_vectors = []
            for word in title_tokens:
                binary_vector = np.zeros(len(vocab))
                if word in vocab:
                    binary_vector[self.token2idx[word]] = 1
                else:
                    binary_vector[self.token2idx[unk_token]] = 1
                title_vectors.append(binary_vector)

            vectors.append(np.array(title_vectors))
        self.data['vectors'] = vectors

        # label genre
        with open('dataset/genres.txt', 'r') as f:
            genre_all = f.readlines()
            genre_all = [x.replace('\n','') for x in genre_all]
        self.genre2idx = {genre:idx for idx, genre in enumerate(genre_all)}

    def tokenize(self, text):
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        tokens = wordpunct_tokenize(text)
        tokens = tokens[:-1] # remove last token because it is the year which maybe is not useful
        return tokens

    def create_vocab(self):
        df = self.movies_train.copy()
        arr_title = df['title'].tolist()
        vocab = set()
        for title in arr_title:
            tokens = self.tokenize(title)
            vocab.update(tokens)
        vocab = list(vocab)
        pad_token = '<PAD>'
        unk_token = '<UNK>'
        vocab.append(pad_token)
        vocab.append(unk_token)
        return vocab


    def __getitem__(self, index):
        title = self.data.iloc[index].title
        img_path = self.data.iloc[index].img_path
        genre = self.data.iloc[index].genre

        # preprocess text
        title_vector = self.data.iloc[index].vectors
        title_tensor = torch.from_numpy(title_vector).float()

        # preprocess img
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
        else:
            img = np.random.rand(256,256,3)
        img = cv2.resize(img, (256,256))
        img_tensor = torch.from_numpy(img.transpose(2,0,1)).float()

        # preprocess label
        genre_vector = np.zeros(len(self.genre2idx))

        for g in genre:
            genre_vector[self.genre2idx[g]] = 1
        genre_tensor = torch.from_numpy(genre_vector).float()

        return title_tensor, img_tensor, genre_tensor

    def __len__(self):
        return len(self.data)
    