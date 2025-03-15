# 🍳 Cooking Chatbot

## 🌟 Overview

Welcome to **Cooking Chatbot**! This chatbot helps users find delicious recipes, ingredients, cooking tips, and step-by-step instructions. It utilizes **Natural Language Processing (NLP)** and a **deep learning model** built with **PyTorch** to classify user queries and provide relevant recipe information.

## ✨ Features

✅ **Intelligent Recipe Finder** - Enter a dish name, and the chatbot will fetch its ingredients, tips, and cooking steps.  
✅ **Natural Language Understanding** - Uses **NLTK** for tokenization and lemmatization.  
✅ **Customizable Neural Network** - Trained using **PyTorch** with a bag-of-words approach.  
✅ **Interactive Chat Experience** - Available via **CLI** or **Streamlit UI**.  
✅ **Model Training & Storage** - Train your own chatbot and save it for future use.  

## 🛠 Installation

### 🔹 Prerequisites
Ensure you have the following installed:

- Python **3.8+**
- `pip` (Python package manager)
- Virtual environment (optional but recommended)

### 🔹 Install Dependencies

1️⃣ Clone the repository:
```bash
 git clone https://github.com/dylanvu6868/Cooking_Chatbot_Basics.git
 cd Cooking_Chatbot_Basics
```

2️⃣ Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate    # Windows
```

3️⃣ Install required packages:
```bash
pip install -r requirements.txt
```

4️⃣ Ensure **NLTK** data is installed:
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

## 🚀 Usage

### 🔥 Train the Model
If you need to **train the chatbot**, run:
```bash
python train.py
```
This will process `data.json`, train the neural network model, and save it as `chatbot_model.pth`.

### 🤖 Run the Chatbot (CLI)
To start the chatbot in command-line mode:
```bash
python main.py
```
Then, enter a dish name, and the bot will respond with:
- 🍽 **Recipe Name**
- 🛒 **Ingredients**
- 💡 **Cooking Tips**
- 📖 **Step-by-Step Instructions**

To exit, type `/quit`.

### 🎨 Run the Chatbot (Streamlit UI)
For a more **interactive experience**, launch the **Streamlit UI**:
```bash
streamlit run streamlit_app.py
```
You can then interact with the chatbot via a web interface.

## 📁 Project Structure
```
📂 Cooking_Chatbot_Basics
├── 📜 assistant.py          # Core chatbot logic
├── 📜 chatbot_model.pth     # Trained model (generated after training)
├── 📜 data.json             # Recipe dataset
├── 📜 dimensions.json       # Model configuration
├── 📜 main.py               # CLI chatbot interface
├── 📜 model.py              # PyTorch model definition
├── 📜 requirements.txt      # Required dependencies
├── 📜 streamlit_app.py      # Streamlit UI for chatbot
├── 📜 train.py              # Model training script
└── 📜 README.md             # Project documentation
```

## 🧠 Model Details
- **Input Layer**: Bag-of-words representation of user input.
- **Hidden Layers**: Two **fully connected layers** with **ReLU activation**.
- **Output Layer**: Softmax function for **intent classification**.
- **Loss Function**: CrossEntropyLoss.
- **Optimizer**: Adam optimizer.

## ⚠️ Troubleshooting

🔹 **NLTK Import Error**: Ensure `nltk` is installed and required data files (`punkt`, `wordnet`) are downloaded.  
🔹 **Missing Dependencies**: Run `pip install -r requirements.txt` again.  
🔹 **Virtual Environment Issues**: Ensure you activate your virtual environment before running scripts.  

## 👨‍💻 Contributors

- 🏆 **Dylan Vu** ([GitHub](https://github.com/dylanvu6868))
- 📂 **Recipe Data** sourced from Minh Dung Tran
