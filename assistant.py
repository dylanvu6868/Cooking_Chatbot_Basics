import os  # Thư viện để thao tác với file hệ thống
import json  # Thư viện để xử lý dữ liệu JSON
import nltk  # Thư viện xử lý ngôn ngữ tự nhiên (NLP)
import numpy as np  # Thư viện tính toán ma trận, mảng số học
import torch  # Thư viện machine learning với deep learning
from torch.utils.data import DataLoader, TensorDataset  # Quản lý dữ liệu theo batch cho việc huấn luyện mô hình
from model import ChatbotModel  # Import model chatbot (cần tạo file model.py chứa lớp ChatbotModel)

# Tải dữ liệu cần thiết từ NLTK (gói hỗ trợ xử lý văn bản)
nltk.download('punkt')  # Bộ tokenizer để chia nhỏ văn bản thành từng từ
nltk.download('wordnet')  # Bộ công cụ lemmatization giúp chuẩn hóa từ vựng

class ChatbotAssistant:
    def __init__(self, intents_path):
        """
        Khởi tạo chatbot với đường dẫn đến file dữ liệu intents.json.
        """
        self.model = None  # Mô hình sẽ được load hoặc train
        self.intents_path = intents_path  # Đường dẫn đến file intents.json
        self.documents = []  # Lưu trữ danh sách các câu intents đã tokenized
        self.vocabulary = []  # Danh sách từ vựng chatbot có thể nhận diện
        self.intents = []  # Danh sách các intents (món ăn)
        self.intents_responses = {}  # Lưu trữ dữ liệu phản hồi theo intent

        # Dữ liệu huấn luyện
        self.X = None  # Đầu vào (bag of words)
        self.y = None  # Nhãn tương ứng (index của intent)

    @staticmethod
    def tokenize_and_lemmatize(text):
        """
        Chia nhỏ câu thành các từ (tokenization) và chuẩn hóa về dạng cơ bản (lemmatization).
        """
        lemmatizer = nltk.WordNetLemmatizer()  # Tạo bộ lemmatizer
        words = nltk.word_tokenize(text)  # Tokenize văn bản (chuyển thành danh sách từ)
        return [lemmatizer.lemmatize(word.lower()) for word in words]  # Chuyển thành chữ thường và lemmatize

    def bag_of_words(self, words):
        """
        Chuyển danh sách từ thành vector bag-of-words.
        Nếu từ có trong vocabulary -> 1, không có -> 0.
        """
        if not self.vocabulary:
            raise ValueError("[❌ ERROR] Vocabulary is empty! Check data loading.")  # Báo lỗi nếu từ điển trống

        # Biểu diễn input dưới dạng vector bag-of-words
        bag = np.array([1 if word in words else 0 for word in self.vocabulary], dtype=np.float32)

        if len(bag) == 0:  # Trường hợp không có từ nào khớp với vocabulary
            print("[⚠️ Warning] No words matched! Creating default bag.")
            bag = np.zeros(len(self.vocabulary), dtype=np.float32)  # Vector toàn số 0
            bag[0] = 1  # Gán 1 vào vị trí đầu tiên để tránh lỗi khi huấn luyện

        return torch.tensor(bag)  # Chuyển về tensor để dùng trong PyTorch

    def parse_intents(self):
        """
        Đọc file JSON, trích xuất intents, và xây dựng vocabulary.
        """
        with open(self.intents_path, 'r', encoding='utf-8') as f:
            intents_data = json.load(f)  # Đọc dữ liệu từ file JSON

        for intent in intents_data['intents']:  # Lặp qua từng intent trong file JSON
            recipe_name = intent['Recipes']  # Lấy tên món ăn
            self.intents.append(recipe_name)  # Thêm vào danh sách intents
            self.intents_responses[recipe_name] = {  # Lưu thông tin phản hồi
                "Ingredients": intent.get("Ingredients", "Không có thông tin nguyên liệu."),
                "Tip": intent.get("Tip", "Không có mẹo nấu ăn."),
                "Steps": intent.get("Các bước tiến hành", "Không có hướng dẫn."),
            }
            pattern_words = self.tokenize_and_lemmatize(recipe_name)  # Tokenize và lemmatize tên món ăn
            self.vocabulary.extend(pattern_words)  # Thêm từ vào danh sách từ vựng
            self.documents.append((pattern_words, recipe_name))  # Lưu vào danh sách tài liệu

        self.vocabulary = sorted(set(self.vocabulary))  # Loại bỏ trùng lặp và sắp xếp từ vựng
        print(f"Tổng số từ trong vocabulary: {len(self.vocabulary)}")  # Debug

    def prepare_data(self):
        """
        Chuyển dữ liệu intents thành dạng bag-of-words để huấn luyện mô hình.
        """
        bags, indices = [], []

        for document in self.documents:
            bag = self.bag_of_words(document[0]).numpy()  # Chuyển câu thành vector bag-of-words
            intent_index = self.intents.index(document[1])  # Tìm index của intent
            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags, dtype=np.float32)  # Đầu vào X
        self.y = np.array(indices, dtype=np.int64)  # Nhãn y

    def train_model(self, batch_size, lr, epochs):
        """
        Huấn luyện mô hình chatbot bằng mạng neuron đơn giản.
        """
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)  # Tạo dataset
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Tạo DataLoader để huấn luyện

        self.model = ChatbotModel(self.X.shape[1], len(self.intents))  # Khởi tạo mô hình

        criterion = torch.nn.CrossEntropyLoss()  # Hàm mất mát
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)  # Trình tối ưu Adam

        for epoch in range(epochs):  # Vòng lặp huấn luyện
            running_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)  # Dự đoán
                loss = criterion(outputs, batch_y)  # Tính loss
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch+1}: Loss: {running_loss / len(loader):.4f}")  # In loss

    def process_message(self, input_message):
        """
        Xử lý câu đầu vào của người dùng, dự đoán intent, và trả về phản hồi.
        """
        words = self.tokenize_and_lemmatize(input_message)  # Tokenize input
        bag = self.bag_of_words(words).unsqueeze(0)  # Chuyển input thành dạng phù hợp

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag)  # Dự đoán

        predicted_class_index = torch.argmax(predictions, dim=1).item()
        predicted_intent = self.intents[predicted_class_index]

        response_data = self.intents_responses.get(predicted_intent, {})

        return f"""Món ăn: {predicted_intent}
Nguyên liệu: {response_data.get('Ingredients', 'Không có thông tin.')}
Mẹo nấu ăn: {response_data.get('Tip', 'Không có mẹo.')}
Các bước thực hiện: {response_data.get('Steps', 'Không có hướng dẫn.')}"""

