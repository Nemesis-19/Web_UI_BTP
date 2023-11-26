from flask import Flask, render_template, request, jsonify
from flask_ngrok import run_with_ngrok  # Import the ngrok wrapper
import torch
import os

import speechbrain as sb
import sentencepiece as spm

from speechbrain.pretrained import EncoderDecoderASR

# for reproducability
SEED = 34
import tensorflow as tf

tf.random.set_seed(SEED)

from inferencer import InferenceModel

state_dict = torch.load(
    "/content/Web_UI_BTP/norm_inference/model.ckpt"
)
# state_dict = torch.load("/content/speechbrain/templates/speech_recognition/LM/eb_inference/model.ckpt")

# for key, value in state_dict.items():
#   print(key, value.shape)

model = InferenceModel()
model.load_state_dict(state_dict)
model.eval()

# load the sentencepiece model
sp = spm.SentencePieceProcessor()
sp.load(
    "/content/Web_UI_BTP/norm_inference/1000_unigram.model"
)
# sp.load("/content/speechbrain/templates/speech_recognition/LM/eb_inference/1000_unigram.model")
# generate a sentence
# text = "this is a test sentence"
# ids = sp.encode_as_ids(text)
# ids = torch.tensor(ids, dtype=torch.long)
# ids = ids.unsqueeze(0)
# output = model(ids)
# output = output.squeeze(0)
# output = torch.argmax(output, dim=-1)
# print(output)
# output = sp.DecodeIds(output)
# print(output)

# Assume max_length_to_generate and model are already defined
# max_length_to_generate = 300  # You can choose a different value

# get transformers
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

# get large GPT2 tokenizer and GPT2 model
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
# GPT2 = TFGPT2LMHeadModel.from_pretrained(
#     "gpt2-large", pad_token_id=tokenizer.eos_token_id
# )

# tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
# GPT2 = TFGPT2LMHeadModel.from_pretrained("gpt2-medium", pad_token_id=tokenizer.eos_token_id)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
GPT2 = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

# view model parameters
# GPT2.summary()

# Initialize the ASR model
asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="pretrained_models/asr-crdnn-rnnlm-librispeech")

app = Flask(__name__)
run_with_ngrok(app)  # Starts ngrok when the app is run

@app.route('/')
def landing():
    return render_template('landing.html')


@app.route('/text-generation', methods=['GET', 'POST'])
def text_generation():
    seed_tokens = []
    seed_vocab = []
    generated_text = ""
    generated_tokens = []
    generated_vocab = []
    word_gen = 0

    seed_tokens_hf = []
    seed_vocab_hf = []
    generated_text_hf = ""
    generated_tokens_hf = []
    generated_vocab_hf = []
    word_gen_hf = 0

    # Read the vocab file and prepare the vocab items
    vocab_items = []
    with open(
        "/content/Web_UI_BTP/norm_inference/vocab.txt",
        "r",
    ) as f:
        for token_id, line in enumerate(f):
            if line.startswith("#"):
                continue  # Skip comment lines
            parts = line.split()
            if len(parts) >= 2:
                token = parts[0]
                vocab_items.append((token_id, token))

    if request.method == "POST":
        input_paragraph = request.form["input_paragraph"]

        max_length_to_generate = int(request.form.get("num_tokens", 100))

        if input_paragraph == "" or max_length_to_generate == 0:
            return render_template(
                "final_index.html",
                vocab_items=vocab_items,
                seed_tokens=seed_tokens,
                seed_vocab=seed_vocab,
                generated_text=generated_text,
                generated_vocab=generated_vocab,
                generated_tokens=generated_tokens,
                word_gen=word_gen,
                seed_tokens_hf=seed_tokens_hf,
                seed_vocab_hf=seed_vocab_hf,
                generated_text_hf=generated_text_hf,
                generated_tokens_hf=generated_tokens_hf,
                generated_vocab_hf=generated_vocab_hf,
                word_gen_hf=word_gen_hf,
            )

        # Perform your computations here and store them in result1 and result2

        seed_text = input_paragraph.upper()
        seed_tokens = sp.encode_as_ids(seed_text)
        seed_vocab = sp.encode_as_pieces(seed_text)

        # print(seed_text)
        # print(seed_tokens)

        # Convert seed tokens to a PyTorch tensor and add a batch dimension
        # input_tensor = torch.tensor([seed_tokens[-1]], dtype=torch.long).unsqueeze(0)
        # print(input_tensor.shape)
        input_tensor = torch.tensor([seed_tokens], dtype=torch.long)
        # print(input_tensor.shape)

        # print(input_tensor.shape)

        # We will store our generated tokens here, starting with the seed
        generated_tokens = seed_tokens[:]

        # Initialize the hidden state for the first forward pass
        hidden_size = model.rnn.hidden_size
        num_layers = model.rnn.num_layers
        num_directions = 2 if model.rnn.bidirectional else 1

        # Initialize hidden state (h) and cell state (c)
        hidden = (
            torch.zeros(
                num_layers * num_directions, len(seed_tokens), hidden_size
            ),
            torch.zeros(
                num_layers * num_directions, len(seed_tokens), hidden_size
            ),
        )

        # hidden = None
        # Generation loop
        with torch.no_grad():
            for _ in range(max_length_to_generate):
                # Forward pass through the model
                logits, hidden = model(input_tensor, hidden)

                # print(logits.shape)
                # for p in hidden:
                #   print(p.shape)
                #   break

                # hidden = list(hidden)

                # logits, h = model(input_tensor)

                # for p in hidden:
                #   print(p.shape)
                #   break

                # print(logits[-1].shape)

                # print(logits.shape)

                # logits should be 2-dimensional [batch_size, vocab_size]
                # We only use the last output for the next token prediction
                logits = logits[:, -1, :].squeeze(0)

                # print(logits.shape)

                indices_to_remove = (
                    logits < torch.topk(logits, 10)[0][..., -1, None]
                )
                logits[indices_to_remove] = -float("Inf")

                # print(logits.shape)

                # Convert the logits to probabilities
                probabilities = torch.softmax(logits, dim=-1)

                # Sample from the probability distribution
                next_token_id = torch.multinomial(
                    probabilities, num_samples=1
                ).squeeze()
                if next_token_id == 0:
                    # if next_token_id == 0 or next_token_id == 1:
                    next_token_id = torch.argmax(probabilities, dim=-1)

                # Append the predicted token to the sequence
                generated_tokens.append(next_token_id.item())

                # print(input_tensor.shape)

                # Update the input tensor to contain only the new token, preserving batch dimension
                # input_tensor = torch.cat((input_tensor, next_token_id.unsqueeze(0).unsqueeze(0)), dim=1)
                next_token_id = next_token_id.unsqueeze(0).unsqueeze(0)

                # print(next_token_id.shape)
                # print(input_tensor.shape)

                input_tensor = torch.concat(
                    (input_tensor[:, 1:], next_token_id), dim=1
                )

                # print(input_tensor.shape)

                # Check if the end-of-sentence token was generated (assuming you have EOS token id)
                # if next_token_id.item() == sp.eos_id():
                #     break

        # Decode the generated tokens to text
        generated_text = sp.decode_ids(generated_tokens)
        generated_vocab = sp.decode_pieces(generated_tokens)
        # print(generated_tokens)
        # print(generated_text)

        word_gen = len(
            generated_text.split()
        )  # Example computation: count words

        seed_text_hf = input_paragraph
        seed_tokens_hf = tokenizer.encode(seed_text_hf, return_tensors="tf")
        seed_vocab_hf = [
            tokenizer.decode(token_id)
            for token_id in seed_tokens_hf.numpy().tolist()
        ]

        # combine both sampling techniques
        sample_outputs = GPT2.generate(
            seed_tokens_hf,
            do_sample=True,
            max_length=max_length_to_generate,
            # temperature = .7,
            top_k=50,
            top_p=0.85,
            num_return_sequences=1,
        )

        generated_tokens_hf = sample_outputs[0]
        generated_text_hf = tokenizer.decode(
            generated_tokens_hf, skip_special_tokens=True
        ).upper()
        generated_vocab_hf = [
            tokenizer.decode(token_id)
            for token_id in generated_tokens_hf.numpy().tolist()
        ]

        word_gen_hf = len(
            generated_text_hf.split()
        )  # Example computation: count words

    return render_template(
        "lm_index.html",
        vocab_items=vocab_items,
        seed_tokens=seed_tokens,
        seed_vocab=seed_vocab,
        generated_text=generated_text,
        generated_vocab=generated_vocab,
        generated_tokens=generated_tokens,
        word_gen=word_gen,
        seed_tokens_hf=seed_tokens_hf,
        seed_vocab_hf=seed_vocab_hf,
        generated_text_hf=generated_text_hf,
        generated_tokens_hf=generated_tokens_hf,
        generated_vocab_hf=generated_vocab_hf,
        word_gen_hf=word_gen_hf,
    )

@app.route('/asr-inference', methods=['GET', 'POST'])
def asr_inference():
    if request.method == 'POST':
        if 'audio_file' in request.files:
            audio_file = request.files['audio_file']
            file_path = 'uploaded_audio.wav'
            audio_file.save(file_path)

            if os.path.exists(file_path):
                transcription = asr_model.transcribe_file(file_path)
                return jsonify(transcription=transcription)
            else:
                return jsonify(transcription="File not found or is empty")
        else:
            return jsonify(transcription="No audio file received")
    return render_template('asr_index.html')

if __name__ == "__main__":
    app.run()  # The app will be accessible at the ngrok URL
