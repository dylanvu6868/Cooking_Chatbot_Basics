import torch
import torch.nn as nn
import torch.nn.functional as F

class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        """
        Mô hình mạng neural cho chatbot.

        Args:
        - input_size (int): Kích thước đầu vào (số lượng từ trong vocabulary).
        - output_size (int): Số lượng lớp đầu ra (tương ứng với số intent).

        Mô hình gồm:
        - 3 lớp fully connected (fc1, fc2, fc3)
        - Hàm kích hoạt ReLU giúp tăng tính phi tuyến
        """
        super(ChatbotModel, self).__init__()

        # Lớp fully connected đầu tiên: Nhận đầu vào có kích thước input_size và tạo ra 128 neuron
        self.fc1 = nn.Linear(input_size, 128)

        # Lớp fully connected thứ hai: Nhận 128 đầu vào và giảm xuống 64 neuron
        self.fc2 = nn.Linear(128, 64)

        # Lớp fully connected cuối cùng: Nhận 64 đầu vào và xuất ra số lượng lớp output_size (tương ứng với số intent)
        self.fc3 = nn.Linear(64, output_size)

        # Hàm kích hoạt ReLU để tạo tính phi tuyến cho mô hình
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Hàm truyền dữ liệu qua mạng neural.

        Args:
        - x (Tensor): Dữ liệu đầu vào (biểu diễn bag-of-words).

        Returns:
        - Tensor: Đầu ra dự đoán (logits) cho từng intent.
        """
        # Truyền dữ liệu qua lớp fully connected đầu tiên và áp dụng ReLU
        x = self.relu(self.fc1(x))

        # Truyền dữ liệu qua lớp fully connected thứ hai và áp dụng ReLU
        x = self.relu(self.fc2(x))

        # Truyền dữ liệu qua lớp fully connected cuối cùng (không có hàm kích hoạt)
        x = self.fc3(x)

        return x  # Trả về logits, sẽ được sử dụng với CrossEntropyLoss để tính loss
