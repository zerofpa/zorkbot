import subprocess
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime
import tkinter as tk
from tkinter import scrolledtext

# Path to your checkpoint folder
checkpoint_path = "/home/xyro/Desktop/resultbert/checkpoint-177"

# Load the pretrained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)

# Set the model to evaluation mode
model.eval()

# List of Zork commands (verbs) and possible objects
verbs = ["look", "examine", "take", "open", "close", "read", "go", "turn on", "turn off", "attack"]
objects = ["room", "door", "lantern", "map", "sword", "troll", "north", "south", "east", "west", "up", "down"]

# Create a combined list of verb-object pairs
commands = [f"{verb} {obj}" for verb in verbs for obj in objects]

def get_model_command(scenario):
    # Preprocess the input text
    inputs = tokenizer([scenario], return_tensors="pt", padding=True, truncation=True)
    
    # Debug: Print inputs to see the processed input text
    print(f"Processed Inputs: {inputs}")
    
    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Debug: Print outputs to see the model's raw outputs
    print(f"Model Outputs: {outputs.logits}")

    # Get the predicted label
    prediction = torch.argmax(outputs.logits, dim=-1)
    
    # Debug: Print the predicted label index
    print(f"Predicted Label Index: {prediction.item()}")

    # Handle out-of-range index
    if prediction.item() >= len(commands):
        predicted_command = "look around"  # Default command
    else:
        # Convert the predicted label to the corresponding command
        predicted_command = commands[prediction.item()]
    
    # Debug: Print the predicted command
    print(f"Predicted Command: {predicted_command}")
    
    return predicted_command

def interact_with_zork(process, command):
    process.stdin.write(command + '\n')
    process.stdin.flush()
    output = process.stdout.read()
    return output

def play_game():
    initial_scenario = "look around"
    scenario = initial_scenario
    
    # Create a directory to save playthroughs
    save_dir = "./playthroughs"
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate a unique filename based on the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"playthrough_{timestamp}.txt")
    
    # Open the file for writing
    with open(filename, 'w') as f:
        # Start the Zork game process
        process = subprocess.Popen(['frotz', 'zork1.dat'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        
        while True:
            log_text(f"Scenario: {scenario}\n")
            f.write(f"Scenario: {scenario}\n")
            print(f"Scenario: {scenario}")
            command = get_model_command(scenario)
            log_text(f"Model's Command: {command}\n")
            f.write(f"Model's Command: {command}\n")
            print(f"Model's Command: {command}")
            
            # Interact with Zork
            response = interact_with_zork(process, command)
            log_text(f"Game Response: {response}\n")
            f.write(f"Game Response: {response}\n")
            print(f"Game Response: {response}")
            
            # Update the scenario with the game's response
            scenario = response.strip()

            # Exit loop if a specific condition is met (e.g., game over)
            if "Game Over" in response or "game over" in response.lower():
                log_text("Game Over detected. Exiting loop.\n")
                f.write("Game Over detected. Exiting loop.\n")
                print("Game Over detected. Exiting loop.")
                break

        process.terminate()

# Create the main application window
root = tk.Tk()
root.title("Zork Game Bot")

# Create a scrolled text widget
log = scrolledtext.ScrolledText(root, width=100, height=30)
log.pack()

def log_text(text):
    log.insert(tk.END, text)
    log.yview(tk.END)

# Create a button to start the game
start_button = tk.Button(root, text="Start Game", command=play_game)
start_button.pack()

# Run the application
root.mainloop()

