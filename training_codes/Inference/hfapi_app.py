from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
#for reproducability
SEED = 34
import tensorflow as tf
tf.random.set_seed(SEED)

app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok when the app is run

#get transformers
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

#get large GPT2 tokenizer and GPT2 model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
GPT2 = TFGPT2LMHeadModel.from_pretrained("gpt2-large", pad_token_id=tokenizer.eos_token_id)

#tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
#GPT2 = TFGPT2LMHeadModel.from_pretrained("gpt2-medium", pad_token_id=tokenizer.eos_token_id)

#tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#GPT2 = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

#view model parameters
GPT2.summary()

@app.route('/', methods=['GET', 'POST'])
def index():
    generated_text = ''
    word_gen = 0
    seed_tokens = []
    generated_tokens = []
    if request.method == 'POST':
        input_paragraph = request.form['input_paragraph']
        num_tokens = int(request.form.get('num_tokens', 100))

        seed_text = input_paragraph
        seed_tokens = tokenizer.encode(seed_text, return_tensors='tf')

        #combine both sampling techniques
        sample_outputs = GPT2.generate(
                                      seed_tokens,
                                      do_sample = True, 
                                      max_length = num_tokens,
                                      # temperature = .7,
                                      top_k = 50, 
                                      top_p = 0.85, 
                                      num_return_sequences = 5
        )

        generated_tokens = sample_outputs[0]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens = True).upper()
        
        word_gen = len(generated_text.split())  # Example computation: count words

    return render_template('index.html', seed_tokens=seed_tokens, generated_text=generated_text, generated_tokens=generated_tokens, word_gen=word_gen)

if __name__ == '__main__':
    app.run()
