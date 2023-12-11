# Import các thư viện cần thiết
import argparse
import sys
import os
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from joblib import dump, load
import numpy as np
from sklearn.metrics import f1_score
from utils import *
import json

# Hàm để chạy quá trình huấn luyện
def run_train(train_dir, dev_dir, model_dir):
    # Tạo thư mục cho mô hình nếu nó chưa tồn tại
    os.makedirs(model_dir, exist_ok=True)

    # Đường dẫn đến các tệp dữ liệu
    train_file = os.path.join(train_dir, 'train.json')
    dev_file = os.path.join(dev_dir, 'dev.json')

    # Đọc dữ liệu huấn luyện và phát triển
    train_data = pd.read_json(train_file, lines=True)
    dev_data = pd.read_json(dev_file, lines=True)

    # Chuẩn bị dữ liệu cho quá trình huấn luyện
    X_train = train_data.drop('two_year_recid', axis=1)
    Y_train = train_data['two_year_recid']
    X_dev = dev_data.drop('two_year_recid', axis=1)
    Y_dev = dev_data['two_year_recid']

    # Biến đổi dữ liệu và thực hiện feature selection
    X_train = preprocess(X_train, split='train')
    X_dev = preprocess(X_dev, split='validation')

    # Tạo và huấn luyện mô hình
    with open('best_hyperparameters.json') as f:
        params = json.load(f)

    model = XGBClassifier(random_state=0, objective='binary:logistic', **params)
    model.fit(X_train, Y_train)
    print(f1_score(model.predict(X_dev), Y_dev))
    # Lưu mô hình
    model_path = os.path.join(model_dir, 'trained_model.joblib')
    dump(model, model_path)


# Hàm để chạy quá trình dự đoán
def run_predict(model_dir, input_dir, output_path):
    # Đường dẫn đến mô hình và dữ liệu đầu vào
    model_path = os.path.join(model_dir, 'trained_model.joblib')
    input_file = os.path.join(input_dir, 'test.json')

    # Tải mô hình
    model = load(model_path)

    # Đọc dữ liệu kiểm tra
    test_data = pd.read_json(input_file, lines=True)

    # Chuẩn bị dữ liệu kiểm tra
    X_test = test_data
    X_test = preprocess(X_test, split='test')

    # Thực hiện dự đoán
    predictions = model.predict(X_test)

    # Lưu kết quả dự đoán
    pd.DataFrame(predictions, columns=['two_year_recid']).to_json(output_path, orient='records', lines=True)


# Hàm chính để xử lý lệnh từ dòng lệnh
def main():
    # Tạo một parser cho các lệnh từ dòng lệnh
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # Tạo parser cho lệnh 'train'
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--train_dir', type=str)
    parser_train.add_argument('--dev_dir', type=str)
    parser_train.add_argument('--model_dir', type=str)

    # Tạo parser cho lệnh 'predict'
    parser_predict = subparsers.add_parser('predict')
    parser_predict.add_argument('--model_dir', type=str)
    parser_predict.add_argument('--input_dir', type=str)
    parser_predict.add_argument('--output_path', type=str)

    # Xử lý các đối số nhập vào
    args = parser.parse_args()

    # Chọn hành động dựa trên lệnh
    if args.command == 'train':
        run_train(args.train_dir, args.dev_dir, args.model_dir)
    elif args.command == 'predict':
        run_predict(args.model_dir, args.input_dir, args.output_path)
    else:
        parser.print_help()
        sys.exit(1)

# Điểm khởi đầu của chương trình
if __name__ == "__main__":
    main()
