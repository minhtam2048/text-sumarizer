INSTALL_MSG = """
Bart will be released through pip in v 3.0.0, until then use it by installing from source:

git clone git@github.com:huggingface/transformers.git
git checkout d6de6423
cd transformers
pip install -e ".[dev]"

"""
import os
import torch
from nltk.tokenize import sent_tokenize, word_tokenize

try:
    import transformers
    from transformers import T5Tokenizer, T5ForConditionalGeneration
except ImportError:
    raise ImportError(INSTALL_MSG)

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained("C:\\Users\\mmiin\\text-summarizer\\TextSumm\\Summarizer\\ML")

def predict(text, num_beam):
    number_of_sentences = sent_tokenize(text)
    print('number_of_sentences: ', len(number_of_sentences))
    number_of_words = word_tokenize(text)
    print('number_of_words: ', len(number_of_words))
    if (len(number_of_sentences) <= 4 and len(number_of_words) <=40):
        summary_txt = text
    else:
        input_ids = tokenizer.batch_encode_plus([text], truncation=True, max_length=1024, return_tensors='pt')['input_ids'].to(torch_device)
        summary_ids = model.generate(input_ids, 
                             num_beams=num_beam, 
                             max_length=142,
                             early_stopping=True)
        summary_txt = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print('Post----: ', summary_txt)
        print('Number of words in summarized text: ', len(word_tokenize(summary_txt)))
    return summary_txt
