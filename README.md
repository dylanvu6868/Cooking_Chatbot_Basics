# Cooking Chatbot

## Overview

This chatbot is designed to provide cooking recipes based on user input. It uses Natural Language Processing (NLP) and a neural network model built with PyTorch to classify user queries and retrieve relevant recipe information.

## Features

- Tokenizes and lemmatizes user input.
- Uses a bag-of-words model for intent classification.
- Trains a neural network using PyTorch.
- Retrieves recipe names, ingredients, cooking tips, and step-by-step instructions.
- Saves and loads trained models for future use.

## Installation

### Prerequisites

Make sure you have the following installed:

- Python 3.8+
- `pip`
- Virtual environment (optional but recommended)

### Install Dependencies

1. Clone the repository:

   ```bash
   git clone <your-repo-url>
   cd chatbot_advanced
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```

3. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Ensure NLTK data is installed:

   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   ```

## Usage

### Train the Model

If you need to train the model before using it, run the following command:

```bash
python train.py
```

This will process the intents from `data.json`, train the model, and save it as `chatbot_model.pth`.

### Running the Chatbot

To start the chatbot, run:

```bash
python main.py
```

Then, enter a dish name, and the bot will provide:

- Ingredients
- Cooking tips
- Step-by-step instructions

To exit, type `/quit`.

## Project Structure

```
 c"c            # Project documentation
```

## Model Details

- **Input Layer**: Bag-of-words representation of user input.
- **Hidden Layers**: Two fully connected layers with ReLU activation and dropout.
- **Output Layer**: Softmax function for classifying intents.
- **Loss Function**: CrossEntropyLoss.
- **Optimizer**: Adam optimizer.

## Troubleshooting

- **NLTK Import Error**: Ensure that `nltk` is installed and the necessary data files (`punkt`, `wordnet`) are downloaded.
- **Missing Dependencies**: Run `pip install -r requirements.txt`.
- **Virtual Environment Issues**: Activate your virtual environment before running the script.

## Contributors

- **dylanvu6868** (@dylanvu6868)
- data.json from Minh Dung TranTran

## License

This project is licensed under the MIT License.

