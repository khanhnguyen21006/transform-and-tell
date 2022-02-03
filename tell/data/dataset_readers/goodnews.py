import logging
import random
from typing import Dict
import torch
import h5py
import json
import os
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from overrides import overrides
from PIL import Image
from pymongo import MongoClient
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor)
from tqdm import tqdm

from tell.data.fields import ImageField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register('good_news')
class GoodNewsReader(DatasetReader):
    """Read from the GoodNews dataset.
    See the repo README for more instruction on how to download the dataset.
    Parameters
    ----------
    tokenizer : ``Tokenizer``
        We use this ``Tokenizer`` for both the premise and the hypothesis.
        See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``
        We similarly use this for both the premise and the hypothesis.
        See :class:`TokenIndexer`.
    """

    def __init__(self,
                 tokenizer: Tokenizer,
                 token_indexers: Dict[str, TokenIndexer],
                 data_dir: str,
                 lazy: bool = True) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers

        self.data_dir = data_dir
        self.preprocess = Compose([Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        random.seed(1234)

        self.rs = np.random.RandomState(1234)

    @overrides
    def _read(self, split: str):
        # split can be either train, valid, or test
        # validation and test sets contain 10K examples each
        if split not in ['TRAIN', 'VAL', 'TEST']:
            raise ValueError(f'Unknown split: {split}')
        # data_name = 'wit_500_min_word_freq'
        logger.info('Grabbing all article IDs')

        h = h5py.File(os.path.join(self.data_dir, 'TEST' + '_IMAGES.hdf5'), 'r')
        imgs = h['images']

        with open(os.path.join(self.data_dir, split + '_IMAGEIDS.json'), 'r') as j:
            image_ids = json.load(j)
        with open(os.path.join(self.data_dir, split + '_ARTICLEURLS.json'), 'r') as j:
            article_urls = json.load(j)
        with open(os.path.join(self.data_dir, split + '_STRARTICLES.json'), 'r') as j:
            str_articles = json.load(j)
        with open(os.path.join(self.data_dir, split + '_STRCAPS.json'), 'r') as j:
            str_captions = json.load(j)
        with open(os.path.join(self.data_dir, split + '_RAWSTRARTICLES.json'), 'r') as j:
            raw_str_articles = json.load(j)
        with open(os.path.join(self.data_dir, split + '_RAWSTRCAPS.json'), 'r') as j:
            raw_str_captions = json.load(j)

        logger.info('Grabbed...')
        # Total number of datapoints
        dataset_size = len(str_captions)
        ids = list(range(dataset_size))

        self.rs.shuffle(ids)
        logger.info('shuffled...')
        for i in tqdm(ids):
            image_id = image_ids[i]
            image = torch.FloatTensor(imgs[i] / 255.)
            article = str_articles[i]
            caption = str_captions[i]
            article_url = article_urls[i]

            if not article or len(article) == 0:
                continue

            if len(article) > 500:
                article = article[:500]

            if not caption or len(caption) == 0:
                continue

            yield self.entry_to_instance(article, image, caption, image_id, article_url)
        logger.info('done.')

    def entry_to_instance(self, article, image, caption, image_id, article_url) -> Instance:
        context = ' '.join(article).strip()
        caption = ' '.join(caption).strip()

        context_tokens = self._tokenizer.tokenize(context)
        caption_tokens = self._tokenizer.tokenize(caption)

        fields = {
            'context': TextField(context_tokens, self._token_indexers),
            'image': ImageField(image, self.preprocess),
            'caption': TextField(caption_tokens, self._token_indexers),
        }

        metadata = {'context': context,
                    'caption': caption,
                    'image_id': image_id,
                    'article_url': article_url}
        fields['metadata'] = MetadataField(metadata)

        return Instance(fields)
