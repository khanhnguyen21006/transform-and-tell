import re
from typing import Dict, List
import numpy as np
import torch
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers.token import Token
from allennlp.data.vocabulary import Vocabulary
from overrides import overrides
from spacy.tokens import Doc
import spacy # modify
import random # modify

SPACE_NORMALIZER = re.compile(r"\s+")
nlp = spacy.load("en_core_web_lg") # modify

def to_token_ids(sentence, roberta):
    bpe_tokens = roberta.bpe.encode(sentence)
    bpe_tokens = f'<s> {bpe_tokens} </s>'
    words = tokenize_line(bpe_tokens)

    token_ids = []
    for word in words:
        idx = roberta.task.source_dictionary.indices[word]
        token_ids.append(idx)
    return token_ids


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


@TokenIndexer.register("roberta_ner_cad")
class RobertaNERCadTokenIndexer(TokenIndexer[int]):
    def __init__(self,
                 model_name: str = 'roberta-base',
                 namespace: str = 'bpe',
                 legacy: bool = False,
                 start_tokens: List[str] = None,
                 end_tokens: List[str] = None,
                 token_min_padding_length: int = 0,
                 padding_on_right: bool = True,
                 padding_value: int = 1,
                 max_len: int = 512) -> None:
        super().__init__(token_min_padding_length)
        roberta = torch.hub.load('pytorch/fairseq:2f7e3f3323', 'roberta.base')
        self.source_dictionary = roberta.task.source_dictionary
        self.bpe = roberta.bpe.bpe
        self.bpe_legacy = roberta.bpe
        self._added_to_vocabulary = False
        self._namespace = namespace
        self._padding_on_right = padding_on_right
        self._padding_value = padding_value
        self._max_len = max_len
        self.legacy = legacy

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        # If we only use pretrained models, we don't need to do anything here.
        pass

    def _add_encoding_to_vocabulary(self, vocabulary: Vocabulary) -> None:
        for piece, idx in self.source_dictionary.indices.items():
            vocabulary._token_to_index[self._namespace][piece] = idx
            vocabulary._index_to_token[self._namespace][idx] = piece

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str,
                          doc: Doc = None) -> Dict[str, List[int]]:
        if not self._added_to_vocabulary:
            self._add_encoding_to_vocabulary(vocabulary)
            self._added_to_vocabulary = True

        text = ' '.join([token.text for token in tokens])
        if self.legacy:
            indices = self.encode(text, doc)
            copy_masks = []
        else:
            indices, copy_masks, noise_mask, ner_mask = self.encode(text, doc) # modify

        return {
            index_name: indices,
            f'{index_name}_copy_masks': copy_masks,
            f'{index_name}_noise_masks': noise_mask, # modify
            f'{index_name}_ner_masks': ner_mask, # modify
        }

    def encode(self, sentence, doc):
        if self.legacy:
            return self.encode_legacy(sentence)
            
        bpe_tokens, copy_masks, noise_mask, ner_mask = self._byte_pair_encode(sentence, doc) # modify
        sentence = ' '.join(map(str, bpe_tokens))      
        words = tokenize_line(sentence)
        assert len(words) == len(copy_masks) == len(noise_mask) == len(ner_mask) # modify

        # Enforce maximum length constraint
        words = words[:self._max_len - 2]
        copy_masks = copy_masks[:self._max_len - 2]
        noise_mask = noise_mask[:self._max_len - 2] # modify
        ner_mask = ner_mask[:self._max_len - 2] # modify
        words = ['<s>'] + words + ['</s>']
        copy_masks = [0] + copy_masks + [0]
        noise_mask = [0] + noise_mask + [0] # modify
        ner_mask = [0] + ner_mask + [0] # modify

        token_ids = []
        for word in words:
            idx = self.source_dictionary.indices[word]
            token_ids.append(idx)

        return token_ids, copy_masks, noise_mask, ner_mask # modify

    def encode_legacy(self, sentence):
        bpe_sentence = '<s> ' + self.bpe_legacy.encode(sentence) + ' </s>'
        tokens = self.source_dictionary.encode_line(
            bpe_sentence, append_eos=False)
        return tokens.long().tolist()[:self._max_len]

    def _byte_pair_encode(self, text, doc):
        bpe_raw_tokens = []
        bpe_tokens = []
        bpe_copy_masks = []
        bpe_noise_masks = [] # modify
        bpe_ner_masks = [] # modify

        raw_tokens = self.bpe.re.findall(self.bpe.pat, text)
        # e.g.[' Tomas', ' Maier', ',', ' autumn', '/', 'winter', ' 2014', ',', '\n', ' in', 'Milan', '.']

        copy_masks = self.get_entity_mask(raw_tokens, doc)
        # Same length as raw_tokens

        # modify for whole word masking
        mask_percent = 0.15
        rand = np.random.rand(len(raw_tokens))
        # create mask array
        mask_arr = (rand < mask_percent).tolist()
        assert len(mask_arr) == len(raw_tokens)

        text_doc = nlp(text)
        # ners = []       
        # for ent in text_doc.ents:
        #     if ent.label_ in ['PERSON', 'ORG', 'GPE']:
        #         ent_info = {
        #             'start': ent.start_char,
        #             'end': ent.end_char,
        #             'text': ent.text,
        #             'label': ent.label_,
        #         }
        #         ners.append(ent_info)
        ner_masks = self.get_entity_mask(raw_tokens, text_doc)
        assert len(ner_masks) == len(raw_tokens)   
        
        for raw_token, entity_mask, noise_mask, ner_mask in zip(raw_tokens, copy_masks, mask_arr, ner_masks):
            # e.g. raw_token == " Tomas"

            # I guess this step is used so that we can distinguish between
            # the space separator and the space character.
            token = ''.join(self.bpe.byte_encoder[b]
                            for b in raw_token.encode('utf-8'))
            # e.g. token == "ĠTomas"

            token_ids = [self.bpe.encoder[bpe_token]
                         for bpe_token in self.bpe.bpe(token).split(' ')]
            # e.g. token_ids == [6669, 959]

            bpe_raw_tokens.extend(self.bpe.bpe(token).split(' '))
            bpe_tokens.extend(token_ids)

            # modify
            if noise_mask:
                bpe_noise_masks.extend([1] * len(token_ids))
            else:
                bpe_noise_masks.extend([0] * len(token_ids))

            # token_s = text.index(raw_token)
            # token_e = token_s + len(raw_token) - 1
            # is_ner = False
            # for ner in ners:
            #     if token_s > (ner['end'] - 1) or token_e < ner['start']:
            #         continue
            #     else:
            #         if random.random() < 0.8:
            #             is_ner = True
            #         break
                    
            if ner_mask:
                bpe_ner_masks.extend([1] * len(token_ids))               
            else:
                bpe_ner_masks.extend([0] * len(token_ids))

            if entity_mask == 0:
                bpe_copy_masks.extend([0] * len(token_ids))
            else:
                bpe_copy_masks.extend([1] * len(token_ids))

        return bpe_tokens, bpe_copy_masks, bpe_noise_masks, bpe_ner_masks

    def get_entity_mask(self, tokens, doc):
        # We first compute the start and end points for each token.
        # End points are exclusive.
        # e.g. tokens = [' Tomas', ' Maier', ',', ' autumn', '/', 'winter', ' 2014', ',', '\n', ' in', 'Milan', '.']
        starts = []
        ends = []
        current = 0
        for token in tokens:
            starts.append(current)
            current += len(token)
            ends.append(current)

        copy_masks = [0] * len(tokens)

        if doc is None:
            return copy_masks

        # Next we get the character positions of named entities
        for ent in doc.ents:
            if random.random() < 0.8:
                # A token is part of an entity if it lies strictly inside it
                for i, (start, end, token) in enumerate(zip(starts, ends, tokens)):
                    entity_start = ent.start_char
                    if token[0] == ' ':
                        entity_start -= 1
                    entity_end = ent.end_char

                    if start >= entity_start and end <= entity_end and ent.label_ in ['PERSON', 'ORG', 'GPE']:
                        copy_masks[i] = 1
        return copy_masks

    @overrides
    def get_padding_lengths(self, token: int) -> Dict[str, int]:  # pylint: disable=unused-argument
        return {}

    @overrides
    def as_padded_tensor(self,
                         tokens: Dict[str, List[int]],
                         desired_num_tokens: Dict[str, int],
                         padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:  # pylint: disable=unused-argument
        padded_dict: Dict[str, torch.Tensor] = {}
        for key, val in tokens.items():
            if 'copy_masks' in key:
                def default_value(): return -1
            elif 'noise_masks' in key:
                def default_value(): return 0 # modify
            elif 'ner_masks' in key:
                def default_value(): return 0 # modify
            else:
                def default_value(): return self._padding_value
            padded_val = pad_sequence_to_length(sequence=val,
                                                desired_length=desired_num_tokens[key],
                                                default_value=default_value,
                                                padding_on_right=self._padding_on_right)
            padded_dict[key] = torch.LongTensor(padded_val)
        return padded_dict

    @overrides
    def get_keys(self, index_name: str) -> List[str]:
        """
        We need to override this because the indexer generates multiple keys.
        """
        # pylint: disable=no-self-use
        return [index_name, f'{index_name}_copy_masks']
