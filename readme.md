
# Web Interface for Text Generation using [SpeechBrain](https://github.com/speechbrain/speechbrain)

The Web UI implements SpeechBrain's custom Recurrent Neural Network based Language Model.

## Language Model

The LM is pretrained and its weights are stored in **model.ckpt** file.

The code for Custom Model, along with its YAML and training script can be found in the **training_codes** folder. It can not be used for Inferencing as it is, a modified version used for Inferencing is present in the **inferencer.py** file.

#### Recurrent Neural Network based LM Architecture

![image](https://drive.google.com/uc?export=view&id=1UB3_3FGCkutWFkQjqSRGZFKfXxNKGn3B)

## Tokenizer Model

Tokenizer is implemented using SentencePiece. The tokenizing scheme followed is *Unigram* encoding.

Tokenizer is also pretrained and the model is stored as **1000_unigram.model** file. Vocab size is 1000 and is stored as **vocab.txt** file, containing encodings for each token in vocab.

---

**Note**: *Both the Language Model and Tokenizer are pretrained on [**Mini LibriSpeech Dataset**](https://www.openslr.org/31/)*

---

## Installation

Install the Repository from GitHub

```bash
! git clone https://github.com/Nemesis-19/Web_UI_BTP.git
%cd /content/Web_UI_BTP/
```

Install the Required Libraries

```bash
! pip install speechbrain
! pip install flask-ngrok flask
! pip install pyngrok==4.1.1
! pip install transformers
```

To Run it, you would also need a [ngrok Authorization token](https://ngrok.com/docs/agent/) (scroll down to Authtokens, on how to generate)

After generating an Authtoken use the following command

```bash
! ngrok authtoken 'your_token'
```

## Running

To run the code, simple run the following command

```bash
! python final_app.py
```

For the first run, it will take some time to Install the GPT Model and Tokenizer on its own

After the Tunnel Connection is Established your Output should look like this:

```bash
 * Serving Flask app 'final_app'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
 * Running on http://e3c9-34-142-180-31.ngrok-free.app
 * Traffic stats available on http://127.0.0.1:4040
```

You need to click on the **ngrok** link given in your output

After reaching thier landing page, click on **Visit Site** to acces the Interface

#### Web Interface should look like this

![image](https://drive.google.com/uc?export=view&id=1fOR_pXjsBxqgOr-jvOG_dUXmepVQdP1D)

## How to Use the Interface

Vocabulary Tokens:

- The Interface on its left displays the Vocabulary used by SpeechBrain's Custom RNN based Language Model.
- It contains 1000 tokens, along with their Token IDs (used for encoding-decoding purpose)

Paragraph Input:

- Input any sentence you would like the model to build up on
- The LM is currently inferenced for Causal Modeling only

Number of Tokens:

- Enter the Number of **tokens** *(not words)* you want both the Models to Output
- By default the model will output uptill 100 tokens, which can be increased and decreased

---

*Output Generation is done using SpeechBrain's pretrained LM, we show both Input and Output*

*For Reference, we have also included outputs from **[HuggingFace's GPT2](https://huggingface.co/gpt2)** pretrained for Causal Modeling*

```python
# An exmaple code snippet from their website

from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
```

---

Output Block:

- It shows the input tokens & input vocab generated after tokenizing
- Next are the output tokens and output vocab generated after de-tokenizing
- Final Generated Text (concatenated version of Output Vocab) and its Word Count

## Upcoming

1. Important:
- [ ]  Improving Web UI Design, Hosting Web UI
- [ ]  Integrating Automatic Speech Recognition Inferencing

2. Addtional:
- [ ]  Adding More Functions like Downloading results in JSON format
- [ ]  Adding More options for LM (Transformer LM, N-Grams, etc)
- [ ]  Adding More tasks apart from Text Generation like Q&A, Chat support, etc
