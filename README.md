# Telegram Chatbot for Model Training

## Overview

This project is a Telegram bot that trains a model to write similarly to the user based on the conversation in Telegram. The project includes scripts for model training, inference, and the bot itself.

## Features

- **Interactive Training**: The bot interacts with the user in a Telegram chat to collect data and train the model.
- **Real-time Inference**: The trained model can generate text that mimics the user's writing style.
- **Ease of Use**: Simple setup and integration with Telegram.

## Project Structure

- `AI.py`: Script for training the model using collected data.
- `inference.py`: Script for running inference with the trained model.
- `main.py`: Script for the Telegram bot implementation.
- `requirements.txt`: List of dependencies required for the project.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- Telegram account and bot token (from BotFather)
- Required Python packages (see `requirements.txt`)

### Installation

1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up your Telegram bot:
    - Create a new bot using BotFather on Telegram.
    - Obtain the bot token and add it to your environment variables or directly in `bot.py`.

### Usage

1. **Training the Model**:
    - Start the bot by running:
      ```bash
      python bot.py
      ```
    - Interact with the bot in Telegram to provide training data.

2. **Running Inference**:
    - Once the model is trained, run:
      ```bash
      python inference.py
      ```
    - The script will generate text based on the trained model.


