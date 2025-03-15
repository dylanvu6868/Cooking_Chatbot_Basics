from assistant import ChatbotAssistant  # Import lớp ChatbotAssistant từ file assistant.py

if __name__ == '__main__':
    # Khởi tạo ChatbotAssistant với đường dẫn đến file chứa dữ liệu về món ăn
    assistant = ChatbotAssistant('data.json')

    # Phân tích dữ liệu intents từ file JSON
    assistant.parse_intents()

    # Chuẩn bị dữ liệu huấn luyện (chuyển đổi văn bản thành số)
    assistant.prepare_data()

    # Huấn luyện mô hình với các tham số:
    # - batch_size = 8: Mỗi batch sẽ chứa 8 mẫu
    # - lr (learning rate) = 0.001: Tốc độ học của mô hình
    # - epochs = 50: Chạy quá trình huấn luyện 50 lần trên toàn bộ tập dữ liệu
    assistant.train_model(batch_size=8, lr=0.001, epochs=50)

    # Lưu mô hình đã huấn luyện và kích thước input/output vào file
    assistant.save_model('chatbot_model.pth', 'dimensions.json')
