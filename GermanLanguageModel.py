#!/usr/bin/env python
# coding: utf-8

# In[12]:


from fastai2.text.all import *


# In[13]:


BASE = Path("./")
MODEL_PATH = BASE/'pretrained_lm'


# In[14]:


# create raw text files
JSON_PATH = BASE/'dataset_dataset-Kurkuma-GoogleSERP'
TXT_PATH = BASE/'text_files'
#list(JSON_PATH.glob('*.json'))


# In[31]:


import json
for file in list(JSON_PATH.glob('*.json')):
    jf = json.load(file)
    tf = TXT_PATH/(str(file.stem)+'.txt')
    tf.write_text(jf['text'])


# In[4]:


get_kurkuma_files = partial(get_text_files, folders=[str(TXT_PATH)])


# In[5]:


files = get_kurkuma_files(BASE)


# In[38]:


print(files[0].open().read())


# In[15]:


dls_lm = DataBlock(
    blocks=TextBlock.from_folder(BASE, is_lm=True),
    get_items=get_text_files, splitter=RandomSplitter(0.1)
).dataloaders(BASE, path=BASE, bs=64, seq_len=80)


# In[41]:


dls_lm.show_batch()


# In[16]:


FILE_LM_ENCODER = MODEL_PATH/'30k-pre-ger'
FILE_ITOS = MODEL_PATH/'30k-pre-ger-itos'
config = awd_lstm_lm_config.copy()
config['n_hid'] = 1150


# In[17]:


learn = language_model_learner(dls_lm, 
                               AWD_LSTM, 
                               config=config, 
                               pretrained_fnames=[FILE_LM_ENCODER, FILE_ITOS],
                               drop_mult=0.3,
                               metrics=[accuracy, Perplexity()]).to_fp16()


# In[9]:


learn.fit_one_cycle(1, 2e-2)


# In[10]:


learn.save('1epoch')


# In[12]:


learn.freeze_to(-1)


# In[13]:


learn.fit_one_cycle(6, 2e-2)


# In[14]:


learn.save('1epoch-freeze-1-6epochs')


# In[18]:


learn.load('1epoch-freeze-1-6epochs')


# In[19]:


learn.freeze_to(-2)


# In[21]:


learn.fit_one_cycle(6, slice(1e-5, 1e-3))


# In[22]:


learn.save('1epoch-freeze-2-6epochs')


# In[23]:


TEXT = "Kurkuma ist für die Gesundheit wichtig, weil"
N_WORDS = 400
N_SENTENCES = 2
preds = [learn.predict(TEXT, N_WORDS, temperature=0.75) 
         for _ in range(N_SENTENCES)]


# In[11]:


print("\n".join(preds))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




