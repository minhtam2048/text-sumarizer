import json
import torch
import transformers

import tensorflow_datasets as tfds

from rouge_score import rouge_scorer

def get_rouge(model, tokenizer, key, ds, batch_size = 64, min_length = 210, max_length = 500, device = "cpu", epochs = 1):
    '''Calculates the rouge score of a model on the given dataset

    args
    model: The model to be tested
    tokenizer: tokenizer for the model to be tested
    key: The rouge score we want i.e rouge-1,rouge-2,rouge-L etc.
    ds: dataset from tensorflow.datasets
    batch_size: size of batch to be extracted
    min_length: Minimum length of the output summary
    max_length: Maximum length of the output summary
    device: cuda or cpu

    returns:
    precision: ratio of number of overlapping words in output and reference summary to number of words in output summary
    recall: ratio of number of overlapping words in output and reference summary to number of words in reference summar
    fmeasure: harmonic mean of precision and recall
    '''
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    total_count = 0
    epoch = 0
    key = key
    device = device
    ds_batched = ds.batch(batch_size)
    scorer = rouge_scorer.RougeScorer([key])
    if(device=="cuda"):
        model.cuda()
    print("Starting......")
    for batch in tfds.as_numpy(ds_batched):
        if(epoch==epochs):
          break
        texts,summaries = batch["article"],batch["highlights"]
        step = 0
        for text,summary in zip(texts,summaries):
          preprocessed_txt = str(text).strip().replace("\n","")
          t5_prep = "summarize: "+ preprocessed_txt
          tokenized_text = tokenizer.encode(t5_prep,max_length = len(t5_prep),return_tensors = "pt").to(device)
          summary_ids = model.generate(tokenized_text,num_beams = 4,
                                              no_repeat_ngram_size = 2,
                                              min_length = min_length,
                                              max_length = max_length,
                                              early_stopping = True)
          output = tokenizer.decode(summary_ids[0].to(device), skip_special_tokens = True)
          if(step%10==0):
            print("Step: ",step)
          step += 1
          scores = scorer.score(str(summary),output)
          precision += scores[key].precision
          recall+= scores[key].recall
          f1 += scores[key].fmeasure
        total_count += len(texts)
        print("Average score after, ", total_count, "epochs")
        print("Precision: ",precision/total_count)
        print("Recall: ",recall/total_count)
        print("fmeasure ",f1/total_count)
        print(scores)
        epoch += 1

# Change model and tokenizer here
tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base')
model = transformers.T5ForConditionalGeneration.from_pretrained("C:\\Users\\mmiin\\text-summarizer\\TextSumm\\Summarizer\\ML")

get_rouge(model, tokenizer, "rouge1", ds, batch_size = 128, device = "cuda")

get_rouge(model, tokenizer, "rouge2", ds, batch_size = 128, device = "cuda")

get_rouge(model, tokenizer, "rougeL", ds, batch_size = 128, device = "cuda")