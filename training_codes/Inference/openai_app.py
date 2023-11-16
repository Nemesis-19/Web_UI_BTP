from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import openai

app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok when the app is run

openai.api_key = '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'

@app.route('/', methods=['GET', 'POST'])
def index():
    generated_text = ''
    word_gen = 0
    if request.method == 'POST':
        input_paragraph = request.form['input_paragraph']
        num_tokens = int(request.form.get('num_tokens', 100))

        response = openai.Completion.create(
            engine="davinci",
            prompt=input_paragraph,
            max_tokens=num_tokens,
            n=1,
            stop=None,
            temperature=0.7
        )

        generated_text = response.choices[0].text.strip()
        word_gen = len(generated_text.split())

    return render_template('index.html', generated_text=generated_text, word_gen=word_gen)

if __name__ == '__main__':
    app.run()
