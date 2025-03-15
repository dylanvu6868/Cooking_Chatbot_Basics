import torch
from assistant import ChatbotAssistant  # Import lớp trợ lý chatbot

if __name__ == '__main__':
    # Khởi tạo ChatbotAssistant với đường dẫn đến tệp chứa dữ liệu về món ăn
    assistant = ChatbotAssistant('data.json')

    # Phân tích dữ liệu từ file JSON để lấy danh sách món ăn, nguyên liệu, mẹo nấu ăn, v.v.
    assistant.parse_intents()

    # Tải mô hình đã huấn luyện và thông tin về kích thước đầu vào/đầu ra
    assistant.load_model('chatbot_model.pth', 'dimensions.json')

    # Bắt đầu vòng lặp trò chuyện với người dùng
    while True:
        # Người dùng nhập vào tên món ăn cần tìm
        message = input('Nhập tên món ăn: ')

        # Nếu người dùng nhập '/quit', thoát chương trình
        if message == '/quit':
            break

        # Gửi input của người dùng đến chatbot để xử lý
        response = assistant.process_message(message)

        # In phản hồi từ chatbot
        print(response)
