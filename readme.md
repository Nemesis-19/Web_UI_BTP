
# Web Interface for Text Generation & Automatic Speech Recognition using [SpeechBrain](https://github.com/speechbrain/speechbrain)

The Web UI implements SpeechBrain's custom Recurrent Neural Network-based Language Model for Text Generation & SpeechBrains's custom CRDNN Encoder - Attentional Decoder with Beam Search-based model for Automatic Speech Recognition.

## Language Model

The LM is pretrained, and its weights are stored in the **model.ckpt** file.

The code for Custom Model, along with its YAML and training script, can be found in the **training_codes** folder. It can not be used for Inferencing as it is, a modified version used for Inferencing is present in the **inferencer.py** file.

#### Recurrent Neural Network-based LM Architecture

![image](https://drive.google.com/uc?export=view&id=1UB3_3FGCkutWFkQjqSRGZFKfXxNKGn3B)

## Tokenizer Model

Tokenizer is implemented using SentencePiece. The tokenizing scheme followed is *Unigram* encoding.

Tokenizer is also pretrained, and the model is stored as **1000_unigram.model** file. Vocab size is 1000 and is stored as **vocab.txt** file containing encodings for each token in vocab.

---

**Note**: *Both the Language Model and Tokenizer are pretrained on [**Mini LibriSpeech Dataset**](https://www.openslr.org/31/)*

---

## Automatic Speech Recognition Model

We use SpeechBrain's EncoderDecoder Inference module to generate inferences from an Audio file (.wav file). 

The ASR model is already present on their [HuggingFace terminal](https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech). It is pretrained on entire *[**LibriSpeech Dataset**](https://www.openslr.org/12)*.

It is a plug-n-play ASR model, which means all components, such as the Language Model and tokeniser, are readily available, there is no requirement to train anything from scratch.

#### CNN + RNN + DNN (CRDNN) based ASR Encoder

![image](https://drive.google.com/uc?export=view&id=1ReDmLvtG_KDoSnlCFlWKU9-hIXCi7J5Q)

## Installation

Install the Repository from GitHub

```bash
! git clone https://github.com/Nemesis-19/Web_UI_BTP.git
```

Install the Required Libraries

```bash
! pip install speechbrain
! pip install flask-ngrok flask
! pip install pyngrok==4.1.1
! pip install transformers
```

To Run it, you would also need a [ngrok Authorization token](https://ngrok.com/docs/agent/) (scroll down to Authtokens on how to generate)

After generating an Authtoken, use the following command.

```bash
! ngrok authtoken 'your_token'
```

## Running

To run the code, simply run the following command.

```bash
! python final_app.py
```

For the first run, it will take some time to Install the GPT Model and Tokenizer on its own

After the Tunnel Connection is Established, your Output should look like this:

```bash
 * Serving Flask app 'final_app'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
 * Running on http://e3c9-34-142-180-31.ngrok-free.app
 * Traffic stats available on http://127.0.0.1:4040
```

You need to click on the **ngrok** link given in your output.

After reaching their landing page, click on **Visit Site** to access the Interface.

#### The Web Interface should look like this

![image](https://drive.google.com/uc?export=view&id=1gn6kg0eF2EukFOKDjksZfsWjPGjLfG9e)

## How to Use the Interface

#### Text Generation Interface:

Vocabulary Tokens:

- The Interface on its left displays the Vocabulary used by SpeechBrain's Custom RNN-based Language Model.
- It contains 1000 tokens, along with their Token IDs (used for the encoding-decoding purposes)

Paragraph Input:

- Input any sentence you would like the model to build upon
- The LM is currently inferenced for Causal Modeling only

Number of Tokens:

- Enter the Number of **tokens** *(not words)* you want both the Models to Output
- By default, the model will output up to 100 tokens, which can be increased and decreased

---

*Output Generation is done using SpeechBrain's pretrained LM, we show both Input and Output*

*For Reference, we have also included outputs from **[HuggingFace's GPT2](https://huggingface.co/gpt2)** pretrained for Causal Modeling*

```python
# An example code snippet from their website

from transformers import GPT2Tokenizer, GPT2Model

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
```

---

Output Block:

- It shows the input tokens & input vocab generated after tokenizing
- Next are the output tokens and output vocab generated after de-tokenizing
- Final Generated Text (concatenated version of Output Vocab) and its Word Count

![image](https://drive.google.com/uc?export=view&id=1ut-4wppNdyJ6FxuyLjrBdavi2GeoIHOf)

#### Automatic Speech Recognition Interface:

After clicking the Start Recording button, the recording starts. The Animated Frequency starts generating output on a runtime basis.

After you click on the Stop Recording button, the recording stops. The graph ceases to run. The Audio Playback becomes available to be played.

In Audio PLayback, you can Play-Pause the recording, increase the voice & can also download the recording in .wav format.

After clicking on the Submit for Transcription button, the recording goes to the backend, where inference is generated using a pretrained ASR model.

SpeechBrain's EncoderDecoder Inference module makes it a matter of 2-3 lines to get inference on an Audio File.

```python
# An example code snippet from their website

from speechbrain.pretrained import EncoderDecoderASR

asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="pretrained_models/asr-crdnn-rnnlm-librispeech")
asr_model.transcribe_file('speechbrain/asr-crdnn-rnnlm-librispeech/example.wav')
```

---

Output Block:

- It shows the inference generated from querying the ASR model with your recorded audio (.wav) file

![image](https://drive.google.com/uc?export=view&id=1A2jMNOhsaSKCPWAi98q4hgrDIS69E-TD)

## Upcoming

1. Important:
- [X]  Improving Web UI Design, Hosting Web UI
- [X]  Integrating Automatic Speech Recognition Inferencing

2. Additional:
- [ ]  Adding More Functions like Downloading results in JSON format
- [ ]  Adding More options for LM (Transformer LM, N-Grams, etc)
- [ ]  Adding More tasks apart from Text Generation like Q&A, Chat support, etc
