import re
import torch
from typing import List, Tuple, Union
from ru_text_normalization.constants.constants import DEL_TOKEN


class TextPreprocessor:
    """
    Class for text preprocessing before normalization.
    
    Attributes:
        device (torch.device): Computing device (CPU/GPU)
    """
    
    def __init__(self):
        """Initialize text preprocessor."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def prepare_sequences(df) -> Tuple[List[str], List[str]]:
        """
        Prepare sequences from DataFrame for model training.
        
        Args:
            df: DataFrame with columns 'sentence_id', 'before' and optionally 'after'
            
        Returns:
            Tuple[List[str], List[str]]: Tuple of input and target sequences
        """
        sentences = []
        labels = []

        grouped = df.groupby('sentence_id')

        for _, group in grouped:
            tokens = list(map(str, group['before'].tolist()))
            tokens_row = DEL_TOKEN.join(tokens)
            sentences.append(tokens_row)

            if 'after' in group.columns:
                targets = list(map(str, group['after'].tolist()))
                tokens_row = DEL_TOKEN.join(targets)
                labels.append(tokens_row)

        return sentences, labels

    @staticmethod
    def is_unplain_token(token: str) -> bool:
        """
        Check if token is "unplain" (contains special characters).
        
        Args:
            token (str): Token to check
            
        Returns:
            bool: True if token contains special characters
        """
        latin_letters = re.search(r'[a-zA-Z]', token)
        digits = re.search(r'\d', token)
        fractions = re.search(r'[½⅓⅔¼¾⅕⅖⅗⅘⅙⅚⅐⅛⅜⅝⅞⅑⅒]', token)
        greek_letters = re.search(r'[αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]', token)

        return any([latin_letters, digits, fractions, greek_letters])

    def preprocess_sentence(self, sentence: Union[str, List[str]]) -> str:
        """
        Preprocess sentence for normalization model.
        
        Args:
            sentence (Union[str, List[str]]): Input sentence or list of tokens
            
        Returns:
            str: Preprocessed sentence in model format
        """
        if isinstance(sentence, str):
            if DEL_TOKEN in sentence:
                sentence = sentence.split(DEL_TOKEN)
            else:
                sentence = sentence.split()

        result = '<SC1>'
        count = 0
        for token in sentence:
            if self.is_unplain_token(token):
                result += ' [' + token + ']' + f'<extra_id_{count}>'
                count += 1
            else:
                result += ' ' + token

        return result



