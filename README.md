# Chatbot Project 🤖

A simple chatbot built using **Python, NLTK, and Keras** with a GUI interface.  
This project demonstrates intent classification, response generation, and a user-friendly interface.

---

## 🚀 Features
- Intent classification using deep learning
- GUI interface for user interaction
- Customizable intents (`intents.json`)
- Trained model saved as `.h5` with supporting pickle files

---

## 📂 Project Structure
- `chatbot.py` – Core chatbot logic
- `train_chatbot.py` – Training script
- `chat_gui.py` – GUI interface
- `intents.json` – Intents and responses
- `chatbot_model.h5`, `words.pkl`, `classes.pkl` – Trained model files

---

## 🛠️ How to Run
```bash
# Train the model
python train_chatbot.py

# Run the chatbot GUI
python chat_gui.py
