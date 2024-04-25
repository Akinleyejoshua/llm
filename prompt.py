from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
from keras.utils import pad_sequences
import numpy as np
from utils.model import load_gpt_model

model = load_gpt_model("gpt_model")


with open("dataset.txt", "r", encoding="utf-8") as file:
    text_data = file.readlines()
    
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_data)

total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in text_data:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])

def prompt_gpt(text):
    seed_text = text
    prev = seed_text
    
    if seed_text == "":

        return {
            "msg": "Prompt cannot be empty"
        }
    
    else:

        generated_text = ""
        

        while True:
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
            predicted_probs = model.predict(token_list, verbose=0)[0]
            predicted_index = np.argmax(predicted_probs)
            predicted_word = tokenizer.index_word[predicted_index]
            # Check if the predicted word is not already in the generated text
            if predicted_word not in generated_text:
                if predicted_word == "end":
                    break
                else:
                    seed_text += " " + predicted_word
                    generated_text += predicted_word + " "
            else:
                break
        
        # print("You: ", prev)
        # print("Bot: ", generated_text)
        return generated_text
