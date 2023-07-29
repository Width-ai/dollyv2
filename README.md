# Dollyv2 Exploration and Evaluation

This notebook demonstrates how to use the Dollyv2 Language Model in Google Colab. Dollyv2 is a large language model developed by Databricks and is available in three versions: 3B, 7B, and 12B. The numbers refer to the number of parameters in the model, with 12B being the largest and most powerful. Our full article can be found here: https://www.width.ai/post/dollyv2-large-language-model

## Setup

First, we need to mount our Google Drive to the Colab environment. This allows us to access files stored in our Google Drive directly from the notebook.

```python
from google.colab import drive
drive.mount('/content/drive')
```

Next, we check the specifications of the GPU available in our Colab environment using the `nvidia-smi` command.

```python
!nvidia-smi
```

We then install the necessary libraries: `accelerate`, `transformers`, and `torchinfo`.

```python
!pip install accelerate transformers torchinfo
```

## Dollyv2 3B Model

The 3B model is the smallest of the three and requires 5.7GB of memory to download. It can run on a T4 GPU.

We use the `pipeline` function from the `transformers` library to load the model. We specify the model name, the data type for the torch tensors, and set `trust_remote_code` to `True` to allow the model to run custom code. We also set `device_map` to `"auto"` to automatically compute the most optimized device map.

```python
import torch
from transformers import pipeline

dolly_3b = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16,
                    trust_remote_code=True, device_map="auto")
```

We can then use the `summary` function from the `torchinfo` library to get a summary of the model.

```python
from torchinfo import summary
summary(dolly_3b.model)
```

To generate text with the model, we simply call it with a string prompt.

```python
res = dolly_3b("ELI5 what is attention in neural networks.")
```

The result is a list of dictionaries, each containing a `generated_text` key with the generated text as the value.

```python
print(res[0]['generated_text'])
```

Finally, we delete the model to free up memory.

```python
del dolly_3b
```

## Dollyv2 7B Model

The 7B model is larger and requires 13.8GB of memory to download. It requires an A100 GPU to run.

The process for loading, summarizing, generating text with, and deleting the model is the same as for the 3B model.

## Dollyv2 12B Model

The 12B model is the largest and requires 23.8GB of memory to download. It also requires an A100 GPU to run.

If the model has been previously downloaded and saved to Google Drive, we can copy it to the local environment to save time.

```python
!cp -vr '/content/drive/MyDrive/Colab Notebooks/dollyv2-12b' dolly-12b-local
```

The process for loading, summarizing, generating text with, and deleting the model is the same as for the other models.

## Using LangChain for Conversations with Dollyv2

By default, the Dollyv2 models are stateless, meaning they don't remember previous queries or replies. To overcome this, we can use LangChain's `ConversationChain` to maintain a conversation history.

First, we install the `langchain` library.

```python
!pip install langchain
```

We then create a `ConversationChain` with a `ConversationBufferMemory` to store the conversation history.

```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline

hf_pipeline2 = HuggingFacePipeline(pipeline=dolly_12b_convo)

conv_memory = ConversationBufferMemory()

conversation = ConversationChain(
      llm=hf_pipeline2,
      verbose=True,
      memory=conv_memory
)
```

We can then use the `predict` method of the `ConversationChain` to generate responses that take into account the conversation history.

```python
ret = conversation.predict(input="Hi there!")
```

## Medical Summarization

We can use the `ConversationChain` to summarize medical reports. We load a medical report from a CSV file and use the `predict` method to generate a summary.

```python
import csv
import textwrap

with open('/content/drive/MyDrive/Colab Notebooks/medreports/mtsamples.csv', 'r') as f:
  reader = csv.reader(f)
  hdr = next(reader)
  row = next(reader)
  med_report
