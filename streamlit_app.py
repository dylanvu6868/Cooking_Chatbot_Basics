import streamlit as st
import torch
from assistant import ChatbotAssistant  # Import lớp trợ lý chatbot

# Khởi tạo ChatbotAssistant với đường dẫn đến file chứa intents
assistant = ChatbotAssistant("data.json")

# Tải mô hình đã được huấn luyện và thông tin về kích thước input/output
assistant.load_model("chatbot_model.pth", "dimensions.json")

# Giao diện Streamlit
st.title("🍳 Cooking Chatbot")  # Tiêu đề của ứng dụng
st.write("Ask me about recipes, ingredients, and cooking tips!")  # Mô tả ngắn về chatbot

# Ô nhập văn bản cho người dùng
user_input = st.text_input("You:", "")  # Người dùng nhập câu hỏi vào đây

# Nếu có input từ người dùng, xử lý câu hỏi và hiển thị phản hồi từ chatbot
if user_input:
    response = assistant.process_message(user_input)  # Gửi input đến chatbot để xử lý
    st.write("🤖 Bot:", response)  # Hiển thị phản hồi từ chatbot

# Chạy ứng dụng Streamlit bằng lệnh: `streamlit run streamlit_app.py`