import json
import os
import re
import torch

from ..constants.constants import DEL_TOKEN
from ..constants.vebratim_dict import vebratim_dict
from ..utils.preprocessing import TextPreprocessor


class TextPostprocessor:
    """
    Class for postprocessing model normalization results.
    
    Attributes:
        device (torch.device): Computing device (CPU/GPU)
        preprocessor (TextPreprocessor): Text preprocessor instance
        abbreviations_set (set): Set of abbreviations
    """
    
    def __init__(self):
        """Initialize text postprocessor."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocessor = TextPreprocessor()
        self.abbreviations_set = self._load_abbreviations()

    @staticmethod
    def _load_abbreviations() -> set:
        """
        Load list of abbreviations from file.
        
        Returns:
            set: Set of abbreviations
        """
        abbreviations_set = set()
        try:
            with open(os.path.join('../data', 'abbreviations.json'), 'r', encoding='utf-8') as f:
                abbreviations_set = {x.strip() for x in json.load(f) if x.strip()}
        except Exception as e:
            print(f"Error loading abbreviations: {e}")
        return abbreviations_set

    def get_model_output(self, model, tokenizer, input_text: str) -> str:
        """
        Get model output for input text.
        
        Args:
            model: Generation model
            tokenizer: Tokenizer
            input_text (str): Input text
            
        Returns:
            str: Generated text
        """
        input_ids = torch.tensor([tokenizer.encode(input_text)]).to(self.device)
        outputs = model.generate(input_ids, eos_token_id=tokenizer.eos_token_id, early_stopping=True)
        return tokenizer.decode(outputs[0][1:])

    @staticmethod
    def is_roman_numeral(text: str) -> bool:
        """
        Check if text is a Roman numeral.
        
        Args:
            text (str): Text to check
            
        Returns:
            bool: True if text is a Roman numeral
        """
        pattern = r'^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'
        return bool(re.fullmatch(pattern, text))

    def is_english(self, text: str) -> bool:
        """
        Check if text is an English word.
        
        Args:
            text (str): Text to check
            
        Returns:
            bool: True if text is an English word
        """
        return bool(re.fullmatch(r'^[a-zA-Z]+$', text)) and not self.is_roman_numeral(text)

    @staticmethod
    def is_domain_or_ip(text: str) -> bool:
        """
        Check if text is a domain or IP address.
        
        Args:
            text (str): Text to check
            
        Returns:
            bool: True if text is a domain or IP address
        """
        s_clean = re.sub(r'^https?://([^/]+).*$', r'\1', text.strip())
        ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        domain_pattern = r'^([a-zA-Z0-9-]+\.)*[a-zA-Z0-9-]+\.[a-zA-Z]{2,}$'
        return bool(re.fullmatch(ip_pattern, s_clean) or re.fullmatch(domain_pattern, s_clean))

    @staticmethod
    def is_initials(text: str) -> bool:
        """
        Check if text is initials.
        
        Args:
            text (str): Text to check
            
        Returns:
            bool: True if text is initials
        """
        pattern = r'^([А-ЯA-ZЁ]\.)([\s-]?[А-ЯA-ZЁ]\.){0,5}$'
        return bool(re.fullmatch(pattern, text.strip()))

    def is_abbreviation(self, text: str) -> bool:
        """
        Check if text is an abbreviation.
        
        Args:
            text (str): Text to check
            
        Returns:
            bool: True if text is an abbreviation
        """
        if not text.isalpha() or self.is_roman_numeral(text):
            return False
        return text in self.abbreviations_set or (2 <= len(text) <= 5 and text.upper() == text)

    def postprocess_token(self, token: str, processed_token: str) -> str:
        """
        Postprocess single token.
        
        Args:
            token (str): Original token
            processed_token (str): Processed token
            
        Returns:
            str: Final processed token
        """
        result_token = processed_token

        if self.is_english(token):
            result_token = ' '.join(tuple(map(lambda x: x + '_trans', [x for x in processed_token])))

        if self.is_domain_or_ip(token):
            result_token = ' '.join(tuple(map(
                lambda x: x + '_trans' if (x != '.' and x != '/') else 'точка' if (x == '.') else 'косая черта',
                [x for x in processed_token])))

        if self.is_initials(token):
            result_token = ' '.join(token.lower().replace(' ', '').replace('ё', 'е').split('.'))[:-1]

        if self.is_abbreviation(token):
            result_token = ' '.join(token.lower().replace(' ', '').replace('ё', 'е'))

        if token == 'NaN':
            result_token = 'n a'

        if token in vebratim_dict:
            result_token = vebratim_dict[token]

        return result_token

    def postprocess_outputs(self, sentence: str, output: str) -> str:
        """
        Postprocess model outputs.
        
        Args:
            sentence (str): Original sentence
            output (str): Model output
            
        Returns:
            str: Processed sentence
        """
        input_text = self.preprocessor.preprocess_sentence(sentence)
        tokens = sentence.split(DEL_TOKEN)
        result = input_text

        try:
            output_pairs = [line.replace('</s>', '').split('  ') for line in output.split('\n') if line]

            if len(output_pairs[-1]) != 2:
                output_pairs = output_pairs[:-1]

            replacements = {pair[0].strip(): pair[1].strip() for pair in output_pairs}

            pattern = r'\[([^\]]+)\]<extra_id_\d+>'
            matches = re.findall(pattern, input_text)

            result_tokens = []
            replacement_index = 0

            for token in tokens:
                processed_token = token
                result_token = processed_token

                if replacement_index < len(matches) and token.strip() == matches[replacement_index].strip():
                    extra_id = f'<extra_id_{replacement_index}>'
                    if extra_id in replacements:
                        processed_token = replacements[extra_id]
                        replacement_index += 1

                result_token = self.postprocess_token(token, processed_token)
                result_tokens.append(result_token)

            result = DEL_TOKEN.join(result_tokens)
        except:
            result_tokens = []
            for token in tokens:
                result_token = self.postprocess_token(token, token)
                result_tokens.append(result_token)
            result = DEL_TOKEN.join(result_tokens)

        return result

    def normalize_text(self, model, tokenizer, sentence: str) -> str:
        """
        Complete text normalization.
        
        Args:
            model: Normalization model
            tokenizer: Tokenizer
            sentence (str): Input sentence
            
        Returns:
            str: Normalized sentence
        """
        input_text = self.preprocessor.preprocess_sentence(sentence)
        output = self.get_model_output(model, tokenizer, input_text)
        return self.postprocess_outputs(sentence, output)

