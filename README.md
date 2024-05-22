Zork Game Bot with BERT Model

This Python script uses a pre-trained BERT model to play the text-based adventure game Zork. The bot interacts with the game by generating commands based on the game's scenario, attempting to explore and gather information by associating verbs with objects. The entire gameplay is visualized through a tkinter GUI, allowing real-time monitoring of the bot's actions and the game's responses.
Features

    - Uses a BERT model to predict game commands.
    - Associates verbs with objects to form meaningful game commands.
    - Continuously interacts with the game until a "Game Over" condition is met.
    - Displays real-time interaction logs in a tkinter GUI.
    - Saves each playthrough to a text file for future reference.

Prerequisites

    - Python 3.6 or higher
    - Required Python packages: transformers, torch, tkinter
    - Frotz installed for running Zork games (sudo apt-get install frotz on Debian-based systems)
    - Zork1 data file (zork1.dat)
