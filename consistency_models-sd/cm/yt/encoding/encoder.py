from typing import List
from dally.src.data.processing.processor import Preprocessor
from dally.src.inference.utils import dict_of_lists_to_list_of_dicts
from typing import Dict, Any, List


class Encoder(Preprocessor):
    def __call__(self, rows: Dict[str, List[Any]]) -> List[Dict[str, bytes]]:
        result = []
        list_of_rows = dict_of_lists_to_list_of_dicts(rows)
        for row in list_of_rows:
            processed_row = {}
            for preprocess in self.preprocesses:
                processed_row.update(self._process(row, **preprocess))
            result.append(processed_row)
        return result
