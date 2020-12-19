import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from transformers import T5Tokenizer, TFT5ForConditionalGeneration
import datetime
import os
import gc

train_data, info = tfds.load('cnn_dailymail', split = 'train', data_dir = 'data/', with_info=True)
test_data = tfds.load('cnn_dailymail', split = 'test', data_dir = 'data/')

class T5Model(TFT5ForConditionalGeneration):
    def __init__(self, *args, log_dir=None, cache_dir= None, **kwargs):
        super().__init__(*args, **kwargs)
    
    @tf.function
    def train_step(self, data):
        x = data[0]
        y = x['labels']
        with tf.GradientTape() as tape:
            outputs = self(inputs = x['inputs'], attention_mask = x['attention_mask'], labels = y, training=True, return_dict=True)
            loss = outputs.loss
            logits = outputs.logits
            loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, self.trainable_variables)
            
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables)) 
        self.compiled_metrics.update_state(y, logits)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics.update({'loss': loss})
        
        return metrics

    def test_step(self, data):
        x = data[0]
        y = x['labels']
        output = self(inputs = x['inputs'], attention_mask = x['attention_mask'], labels = y, training=False, return_dict=True)
        loss = output.loss
        logits = output.logits
        loss = tf.reduce_mean(loss)
        self.compiled_metrics.update_state(y, logits)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics.update({'loss': loss})
        return metrics

model_path = 'drive/My Drive/Doc_Sum/Trained/models'
log_path = './t5drive/My Drive/Doc_Sum/Trained/logs'
config = {
    'batch_size' : 16,
    'epochs' : 1,
    'learning_rate' :1e-4,
    'max_len' : 512,
    'summary_len' : 150
}
data_size = {
    'train': 100000,
    'test' : 10000
}
params = {'source_len' : 512,
          'target_len' : 150,
          'batch_size' : 4
          }
tokenizer = T5Tokenizer.from_pretrained("t5-small")

class DataGenerator(tf.keras.utils.Sequence):
  def  __init__(self, data, tokenizer, mode, source_len, target_len, batch_size):
   self.data = data
   self.tokenizer = tokenizer
   self.source_len = source_len
   self.target_len = target_len
   self.batch_size = batch_size
   self.mode = mode

  def __len__(self):
     return int(np.ceil(data_size[self.mode]/self.batch_size))
    
  def __getitem__(self, index):
    dataset_batch = self.data.skip(self.batch_size).take(self.batch_size)
    encoded_batch = self.encode_data(dataset_batch, self.tokenizer, self.source_len , self.target_len)
    return encoded_batch

  def encode_data(self, data, tokenizer, source_len , target_len ):
    source = data.map(lambda text: text['article'])
    source = list(map(lambda text: str(text,'utf-8'),list(tfds.as_numpy(source))))
    target = data.map(lambda text: text['highlights'])
    target = list(map(lambda text: str(text,'utf-8'),list(tfds.as_numpy(target))))

    batch_encoding = tokenizer.prepare_seq2seq_batch(
        src_texts = source,
        tgt_texts =  target,
        max_length= source_len, 
        max_target_length= target_len,
        padding = 'max_length',
        return_tensors = 'tf')
    batch_encoding['labels'] = tf.where(batch_encoding['labels']==tokenizer.pad_token_id, -100, batch_encoding['labels'])

    return {'inputs' : batch_encoding['input_ids'], 
            'attention_mask' : batch_encoding['attention_mask'],
            'labels':batch_encoding['labels']
            }

training_data = DataGenerator(train_data.take(data_size['train']), tokenizer, 'train', **params)
validation_data = DataGenerator(test_data.take(data_size['test']), tokenizer, 'test', **params)

log_dir = log_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
                                                     
checkpoint_filepath = model_path + "/" + "T5-{epoch:04d}-{val_loss:.4f}.ckpt"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

callbacks = [tensorboard_callback, model_checkpoint_callback]

def create_model():
  model = T5Model.from_pretrained("t5-small")
  optimizer = tf.keras.optimizers.Adam(lr=config['learning_rate'])
  metrics = tf.keras.metrics.SparseCategoricalAccuracy(name = 'accuracy')
  model.compile(optimizer=optimizer, metrics = metrics)
  return model
model = create_model()


epochs = config['epochs']
model.fit( training_data, validation_data = validation_data, epochs = epochs)

model.save_pretrained(model_path)

LONG_BORING_CNN_NEWS= """Scientists Ugur Sahin and Ozlem Tureci have dedicated their lives to the field of oncology and infectious diseases, and spent years pioneering personalized immunotherapy treatments for cancer. 
But amid the coronavirus pandemic, the couple's groundbreaking research in the field of modified genetic code has catapulted them into the public eye, as the brains behind the world's first effective coronavirus vaccine.
Sahin, 55, and Tureci, 53, set up BioNTech in the central German city of Mainz in 2008. On Monday the company's partner, US pharmaceutical giant Pfizer, said their candidate vaccine was more than 90% effective in preventing infection in volunteers.
It uses the never-before-approved technology called messenger RNA, or mRNA, to spark an immune response in people who are vaccinated.
On a call with reporters on Tuesday, Sahin explained the significance of the news -- and sent a message of hope for the world.
"I think the good message for mankind is that we now understand that COVID-19 infections can be indeed prevented by a vaccine," he said. 
Speaking to CNN on Monday, Pfizer CEO's Albert Bourla called it "the greatest medical advance" in the last 100 years.
While the vaccine is a huge step for the scientific community, Sahin and Tureci are veterans in the world of medical achievements.
The pair, both trained physicians, established their previous company, Ganymed Pharmaceuticals, in 2001 to work on developing cancer-fighting antibodies, eventually selling it for $1.4 billion in 2016.
Chief Executive Sahin and Chief Medical Officer Tureci are listed among Germany's 100 richest people, according to the weekly Welt am Sonntag newspaper. On Tuesday, the market value of their Nasdaq-listed company jumped to $25.72 billion -- a massive leap from $4.6 billion last year.
But the couple's charitable ethos and longstanding commitment to academia and science appear to have kept them grounded, even as their work on the Covid-19 vaccine propels them into the global spotlight.
In May, the couple told CNN they felt compelled to "provide something for society," given the work they had done in their field over the last two decades.
Sahin was born in Iskenderun, a city on Turkey's Mediterranean coast. He moved to Cologne, Germany when he was four, where his father worked at a local Ford factory, according to Reuters.
He met Tureci, the daughter of a Turkish physician, when the pair were both embarking on their academic careers.
Sahin and Tureci bonded over a shared passion for cancer research, according to Reuters, who reported that the couple even began their wedding day in the research lab.
In January, after reading a scientific paper about the coronavirus in Wuhan, China, Sahin was taken by the "small step" from anti-cancer mRNA drugs to mRNA-based viral vaccines, Reuters reported.
BioNTech assigned 500 of its staff to work on the project with several potential mRNA compounds, eventually closing a partnership with Pfizer in March.
Their Covid-19 vaccine approach uses genetic material, mRNA, to trick cells into producing bits of protein that look like pieces of the virus. The immune system learns to recognize and attack those bits and, in theory, would react fast to any actual infection. """

# Load model
model_predict = create_model() 
model_predict.load_weights("drive/My Drive/Doc_Sum/Trained/models/tf_model.h5")

source = tokenizer.prepare_seq2seq_batch(
      src_texts = LONG_BORING_CNN_NEWS,
      max_length=512, 
      padding = 'max_length',
      return_tensors = 'tf')
input_ids = source['input_ids']
attention_mask = source['attention_mask']

summary = model.generate(input_ids = input_ids,
                attention_mask = attention_mask, 
                max_length=150, 
                num_beams=4,
                
                early_stopping=True)                          
decoded_summary = tokenizer.decode(summary.numpy()[0])
print ('Summary:\n', decoded_summary)