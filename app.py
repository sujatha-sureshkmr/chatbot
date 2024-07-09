from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import random
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import re

app = Flask(__name__)
socketio = SocketIO(app)

# Example responses for demonstration

# Function to read relevant keywords from a text file
def read_keywords_from_file(file_path):
    with open(file_path, 'r') as f:
        keywords = [line.strip() for line in f.readlines()]
    return keywords

# Function to check if a question is relevant based on keywords
def is_question_relevant(question, relevant_keywords):
    pattern = r'[^\w\s]'  # Matches any non-word and non-space characters
    # Replace the special characters with an empty string
    cleaned_string = re.sub(pattern, '', question)
    print('cleaned_string',cleaned_string)
    question_tokens = cleaned_string.lower().split()
    for keyword in relevant_keywords:
        if keyword.lower() in question_tokens:
            return True
    return False

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('user_message')
def handle_message(data):
    user_message = data['message']

    # Load fine-tuned model and tokenizer configuration
    model_path = 'fine_tuned_model.pth'  # Update with the path to your fine-tuned model .pth file
    keywords_file = 'EVB_Nissan_2013_relevant_words.txt'  # Path to the relevant keywords text file
    
    # Load tokenizer configuration
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Load model configuration
    config = GPT2Config.from_pretrained('gpt2')
    model = GPT2LMHeadModel(config)

    # Load the state_dict from .pth file
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    # Load relevant keywords from text file
    relevant_keywords = read_keywords_from_file(keywords_file)

    # Example: Generate text
    input_text = user_message
    
    # Example: Check if the question is relevant
    if is_question_relevant(input_text, relevant_keywords):
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        generated = model.generate(input_ids, max_length=100, num_return_sequences=1)
        decoded_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(decoded_text)
    else:
        decoded_text = "Sorry, I am not trained to answer this question.Question is not relevant."

    response = decoded_text
    print(response)
    emit('bot_response', {'message': response})

if __name__ == '__main__':
    socketio.run(app, debug=True)