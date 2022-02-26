from typing import Dict, List, Tuple

from allennlp.data.fields import TextField
from allennlp.data.token_indexers.token_indexer import TokenIndexer, TokenType
from allennlp.data.tokenizers.token import Token
from allennlp.data.vocabulary import Vocabulary
from overrides import overrides

TokenList = List[TokenType]  # pylint: disable=invalid-name


class NERCaptionField(TextField):
    def __init__(self,
                 tokens: List[Token],
                 token_indexers: Dict[str, TokenIndexer],
                 caption: bool = False
                 ) -> None:
        super().__init__(tokens, token_indexers)
        self.caption = caption

    @overrides
    def index(self, vocab: Vocabulary):
        token_arrays: Dict[str, TokenList] = {}
        indexer_name_to_indexed_token: Dict[str, List[str]] = {}
        token_index_to_indexer_name: Dict[str, str] = {}
        for indexer_name, indexer in self._token_indexers.items():
            token_indices = indexer.tokens_to_indices(
                self.tokens, vocab, indexer_name, self.caption)
            token_arrays.update(token_indices)
            indexer_name_to_indexed_token[indexer_name] = list(
                token_indices.keys())
            for token_index in token_indices:
                token_index_to_indexer_name[token_index] = indexer_name
        self._indexed_tokens = token_arrays
        self._indexer_name_to_indexed_token = indexer_name_to_indexed_token
        self._token_index_to_indexer_name = token_index_to_indexer_name