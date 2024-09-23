import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import StackingRegressor
from Mesureregression import NSE
import pickle

# Đọc và tiền xử lý dữ liệu
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :3]  # Các đặc trưng (features)
    y = data.iloc[:, 3]   # Biến mục tiêu (target)

    # Chia tập dữ liệu thành 80% train và 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Sau khi huấn luyện mô hình
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)


    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Khởi tạo và huấn luyện các mô hình
def train_models(X_train, y_train):
    models = {
        'linear': LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'mlp': MLPRegressor(max_iter=1000)
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models

# Tạo mô hình stacking
def create_stacking_model(models):
    base_estimators = [
        ('linear', models['linear']),
        ('ridge', models['ridge']),
        ('mlp', models['mlp'])
    ]
    
    # Meta-model: Ridge
    stacking_model = StackingRegressor(
        estimators=base_estimators,
        final_estimator=Ridge(alpha=1.0)
    )
    
    # Huấn luyện mô hình stacking
    stacking_model.fit(X_train_scaled, y_train)
    
    return stacking_model

# Huấn luyện với Stacking
def train_models_with_stacking(X_train, y_train):
    models = train_models(X_train, y_train)  # Huấn luyện các mô hình cơ bản
    models['stacking'] = create_stacking_model(models)  # Thêm mô hình stacking
    
    # Lưu từng mô hình đã huấn luyện
    for model_name, model in models.items():
        with open(f'{model_name}_model.pkl', 'wb') as f:  # Mở file để ghi nhị phân
            pickle.dump(model, f)  # Lưu mô hình
    
    return models


# Dự đoán từ mô hình đã huấn luyện
def predict_model(models, model_choice, new_data, scaler):
    new_data_scaled = scaler.transform(new_data)

    if model_choice == 'linear':
        model = models['linear']
    elif model_choice == 'ridge':
        model = models['ridge']
    elif model_choice == 'stacking':
        model = models['stacking']
    else:
        model = models['mlp']

    prediction = model.predict(new_data_scaled)

    return prediction

# Đánh giá mô hình
def evaluate_model(models, X_test, y_test):
    evaluations = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        nse = NSE(y_test, y_pred)
        evaluations[name] = {
            'R^2': r2,
            'MAE': mae,
            'RMSE': rmse,
            'NSE':nse
        }
    
    return evaluations

# Hàm vẽ biểu đồ giữa giá trị thực tế và dự đoán của từng mô hình
def plot_actual_vs_predicted(y_test, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)  # Đường cho dự đoán hoàn hảo
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title(f'Predicted vs Actual Sales for {model_name}')
    plt.grid(True)
    plt.show()

# Đường dẫn tới file CSV chứa dữ liệu (thay đổi tùy vào vị trí dữ liệu của bạn)
file_path = 'advertisite.csv'

# Bước 1: Tải và tiền xử lý dữ liệu
X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_and_preprocess_data(file_path)

# Bước 2: Huấn luyện mô hình với Stacking
models = train_models_with_stacking(X_train_scaled, y_train)

# Bước 3: Đánh giá mô hình trên tập kiểm tra
evaluations = evaluate_model(models, X_test_scaled, y_test)

# Bước 4: In ra kết quả đánh giá các mô hình và vẽ biểu đồ
for model_name, eval_metrics in evaluations.items():
    print(f"Đánh giá mô hình: {model_name}")
    for metric, value in eval_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Dự đoán giá trị cho mô hình hiện tại
    y_pred = models[model_name].predict(X_test_scaled)
    
    # Vẽ biểu đồ dự đoán vs thực tế
    plot_actual_vs_predicted(y_test, y_pred, model_name)
    
    print("-" * 30)




