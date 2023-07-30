#!/usr/bin/env python
# coding: utf-8

# In[4]:


# MPS acceleration is available on MacOS 12.3+
get_ipython().system('pip3 install torch torchvision torchaudio')


# In[5]:


get_ipython().system('pip3 install transformers ipywidgets gradio --upgrade')


# In[6]:


import gradio as gr
from transformers import pipeline


# In[7]:


translation_pipeline = pipeline('translation_en_to_de')


# In[9]:


results = translation_pipeline('I love ice-cream')


# In[10]:


results[0]['translation_text']


# In[11]:


def translate_transformers(from_text):
    results = translation_pipeline(from_text)
    return results[0]['translation_text']


# In[12]:


translate_transformers('My name is Sakshi')


# In[13]:


interface = gr.Interface(fn=translate_transformers, 
                         inputs=gr.inputs.Textbox(lines=2, placeholder='Text to translate'),
                        outputs='text')


# In[15]:


interface.launch()


# In[ ]:




