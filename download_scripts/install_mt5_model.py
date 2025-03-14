from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import os

save_directory = "pretrained_weight"
os.makedirs(save_directory, exist_ok=True)

# Load and save the model
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")

# Save model and tokenizer to the 'pretrained' folder
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"Model downloaded and saved in '{save_directory}'")
