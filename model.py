import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import StackingRegressor
from Mesureregression import NSE
from sklearn.model_selection import GridSearchCV
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
    X_test_scaled = scaler.fit_transform(X_test)

    # lưu lại trình chuẩn hóa
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)


    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def find_best_alpha_ridge(X_train, y_train):
    # Thiết lập dải giá trị alpha để thử nghiệm
    alpha_values = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    
    # Khởi tạo mô hình Ridge
    ridge = Ridge()
    
    # Grid search với cross-validation để tìm giá trị alpha tối ưu
    grid_search = GridSearchCV(estimator=ridge, param_grid=alpha_values, cv=5, scoring='r2')
    
    # Thực hiện tìm kiếm
    grid_search.fit(X_train, y_train)
    
    # Lấy ra giá trị alpha tốt nhất
    best_alpha = grid_search.best_params_['alpha']
    best_score = grid_search.best_score_
    
    print("-" * 30)
    print(f"Best alpha: {best_alpha}")
    print(f"Best cross-validation R^2 score: {best_score:.4f}")
    print("-" * 30)
    
    return best_alpha
def find_best_max_iter_mlp(X_train, y_train):
    # Thiết lập dải giá trị max_iter để thử nghiệm
    max_iter_values = {'max_iter': [200, 500, 1000, 1500, 2000]}
    
    # Khởi tạo mô hình MLPRegressor với các tham số mặc định khác
    mlp = MLPRegressor()
    
    # Grid search với cross-validation để tìm giá trị max_iter tối ưu
    grid_search = GridSearchCV(estimator=mlp, param_grid=max_iter_values, cv=5, scoring='r2')
    
    # Thực hiện tìm kiếm
    grid_search.fit(X_train, y_train)
    
    # Lấy ra giá trị max_iter tốt nhất
    best_max_iter = grid_search.best_params_['max_iter']
    best_score = grid_search.best_score_
    
    print("-" * 30)
    print(f"Best max_iter: {best_max_iter}")
    print(f"Best cross-validation R^2 score: {best_score:.4f}")
    print("-" * 30)
    
    return best_max_iter
# Khởi tạo và huấn luyện các mô hình
def train_models(X_train, y_train):
    best_alpha_ridge = find_best_alpha_ridge(X_train_scaled, y_train)
    best_max_iter_mlp = find_best_max_iter_mlp(X_train_scaled, y_train)
    models = {
        'linear': LinearRegression(),
        'ridge': Ridge(alpha=best_alpha_ridge),
        'mlp': MLPRegressor(max_iter=best_max_iter_mlp)
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
        final_estimator=Ridge(alpha=1)
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

# Đánh giá mô hình với cross-validation
def evaluate_model_with_cv(models, X_train, y_train):
    evaluations = {}
    
    for name, model in models.items():
        scoring = {
            'R^2': 'r2',
            'MAE': 'neg_mean_absolute_error',
            'RMSE': 'neg_root_mean_squared_error'
        }
        
        # Sử dụng cross_validate để tính toán trên nhiều chỉ số
        cv_results = cross_validate(model, X_train, y_train, cv=5, scoring=scoring)
        
        # Tính trung bình cho các chỉ số đo lường
        evaluations[name] = {
            'R^2': np.mean(cv_results['test_R^2']),
            'MAE': -np.mean(cv_results['test_MAE']),  # negate because scores are negative for MAE
            'RMSE': -np.mean(cv_results['test_RMSE'])
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
file_path = 'advertising.csv'

# Bước 1: Tải và tiền xử lý dữ liệu
X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_and_preprocess_data(file_path)

# Bước 2: Huấn luyện mô hình với Stacking
models = train_models_with_stacking(X_train_scaled, y_train)

# Bước 3: Đánh giá mô hình
#evaluations = evaluate_model(models, X_test_scaled, y_test)
evaluations = evaluate_model_with_cv(models, X_train_scaled, y_train)

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




