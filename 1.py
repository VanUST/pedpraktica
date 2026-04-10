import numpy as np
import pandas as pd
import matplotlib
# Use 'Agg' for non-interactive (saving to file only)
# Use 'Qt5Agg' or 'macosx' if you have those libraries installed
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import fetch_california_housing, fetch_openml

# Настройки для академического стиля графиков
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12

# =========================================================
# КЕЙС 1: DML на полусинтетических данных (CA Housing)
# =========================================================
def plot_dml_comparison():
    print("--- ЗАПУСК КЕЙСА 1: DOUBLE MACHINE LEARNING (Полусинтетика CA Housing) ---")
    
    california = fetch_california_housing(as_frame=True)
    df = california.frame
    df = df[df['MedHouseVal'] < 5.0] 
    
    # 1. Признаки (X) - реальные данные
    confounders = ['MedInc', 'AveRooms', 'AveBedrms', 'Latitude', 'Longitude']
    X = df[confounders].values
    
    # 2. Моделируем Treatment (T): "Умный дом". Чем богаче район, тем выше шанс установки
    # Центрируем MedInc, чтобы получить нормальные вероятности
    medinc_centered = df['MedInc'] - df['MedInc'].mean()
    prob_t = 1 / (1 + np.exp(-medinc_centered)) # Логистическая функция от дохода
    T = np.random.binomial(1, prob_t)
    
    # 3. Моделируем целевую переменную (Y) с известным истинным эффектом
    true_ate = 0.50  # Истинный каузальный эффект: +$50k к цене
    Y_base = df['MedHouseVal'].values # Истинная базовая цена
    Y = Y_base + true_ate * T + np.random.normal(0, 0.1, len(df))
    
    n = len(Y)
    naive_boot, dml_boot = [], []
    
    print("Выполняется расчет DML (бутстрап)...")
    for i in range(150): 
        idx = np.random.choice(n, n, replace=True)
        X_b, T_b, Y_b = X[idx], T[idx], Y[idx]
        
        # 1. Наивный подход
        naive_ate = np.mean(Y_b[T_b==1]) - np.mean(Y_b[T_b==0])
        naive_boot.append(naive_ate)
        
        # 2. DML подход
        m_y = RandomForestRegressor(max_depth=5, n_estimators=50, n_jobs=-1).fit(X_b, Y_b)
        Y_res = Y_b - m_y.predict(X_b)
        
        m_t = RandomForestClassifier(max_depth=5, n_estimators=50, n_jobs=-1).fit(X_b, T_b)
        T_res = T_b - m_t.predict_proba(X_b)[:, 1]
        
        dml_ate = LinearRegression(fit_intercept=False).fit(T_res.reshape(-1, 1), Y_res).coef_[0]
        dml_boot.append(dml_ate)

    naive_mean = np.mean(naive_boot)
    dml_mean = np.mean(dml_boot)
    
    print(f"Истинный заложенный эффект: {true_ate:.3f}")
    print(f"Наивная оценка (впитала эффект дохода): {naive_mean:.3f}")
    print(f"Оценка DML (очищенная): {dml_mean:.3f}\n")

    plt.figure(figsize=(10, 5))
    sns.kdeplot(naive_boot, fill=True, color='#e74c3c', alpha=0.5, label=f'Наивная оценка: ~{naive_mean:.2f}')
    sns.kdeplot(dml_boot, fill=True, color='#2ecc71', alpha=0.7, label=f'DML Оценка: ~{dml_mean:.2f}')
    
    # Теперь мы можем обоснованно нарисовать линию истинного эффекта
    plt.axvline(true_ate, color='black', linestyle='--', linewidth=2.5, label=f'Истинный эффект: {true_ate:.2f}')
    
    plt.title('Восстановление истинного каузального эффекта методом DML (CA Housing)')
    plt.xlabel('Оценка эффекта (Влияние на цену в $100k)')
    plt.ylabel('Плотность вероятности')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('01_dml_comparison.png', dpi=300)
    plt.close()

# =========================================================
# КЕЙС 2: Гладкий Conformal Prediction
# =========================================================
def plot_conformal_heteroskedasticity():
    print("--- ЗАПУСК КЕЙСА 2: CONFORMAL PREDICTION (Гладкие границы) ---")
    
    california = fetch_california_housing()
    X_full = california.data[:, 0].reshape(-1, 1) # Доход
    Y_full = california.target
    
    X_train_full, X_test, Y_train_full, Y_test = train_test_split(X_full, Y_full, test_size=0.2, random_state=42)
    X_train, X_calib, Y_train, Y_calib = train_test_split(X_train_full, Y_train_full, test_size=0.3, random_state=42)
    
    # Сильная регуляризация для базовой гладкости модели (min_samples_leaf=250)
    model_mean = RandomForestRegressor(n_estimators=100, min_samples_leaf=250, random_state=42, n_jobs=-1)
    model_mean.fit(X_train, Y_train)
    
    residuals_train = np.abs(Y_train - model_mean.predict(X_train))
    model_var = RandomForestRegressor(n_estimators=100, min_samples_leaf=250, random_state=42, n_jobs=-1)
    model_var.fit(X_train, residuals_train)
    
    Y_pred_test = model_mean.predict(X_test)
    global_rmse = np.sqrt(np.mean((Y_train - model_mean.predict(X_train))**2))
    naive_margin = 1.8 * global_rmse 
    
    calib_preds = model_mean.predict(X_calib)
    calib_vars = model_var.predict(X_calib) + 1e-6
    scores = np.abs(Y_calib - calib_preds) / calib_vars
    
    alpha = 0.05
    n_calib = len(X_calib)
    q_level = np.ceil((n_calib + 1) * (1 - alpha)) / n_calib
    q_hat = np.quantile(scores, q_level)
    
    conformal_margins = q_hat * (model_var.predict(X_test) + 1e-6)
    
    lower_bound_naive = np.maximum(0, Y_pred_test - naive_margin)
    lower_bound_cp = np.maximum(0, Y_pred_test - conformal_margins)
    
    naive_coverage = np.mean((Y_test >= lower_bound_naive) & (Y_test <= Y_pred_test + naive_margin))
    cp_coverage = np.mean((Y_test >= lower_bound_cp) & (Y_test <= Y_pred_test + conformal_margins))

    print(f"Общее покрытие (RMSE): {naive_coverage*100:.2f}%")
    print(f"Общее покрытие (Conformal): {cp_coverage*100:.2f}%\n")

    sort_idx = np.argsort(X_test[:, 0])
    x_sorted = X_test[sort_idx, 0]
    y_test_sorted = Y_test[sort_idx]
    
    # Подготовка данных для графиков
    y_pred_sorted = Y_pred_test[sort_idx]
    lower_naive_sorted = lower_bound_naive[sort_idx]
    upper_naive_sorted = y_pred_sorted + naive_margin
    lower_cp_sorted = lower_bound_cp[sort_idx]
    upper_cp_sorted = y_pred_sorted + conformal_margins[sort_idx]
    
    # Гауссова фильтрация для идеальной визуальной гладкости прайсинг-интервалов
    y_pred_smooth = gaussian_filter1d(y_pred_sorted, sigma=5)
    lower_naive_smooth = np.maximum(0, gaussian_filter1d(lower_naive_sorted, sigma=5))
    upper_naive_smooth = gaussian_filter1d(upper_naive_sorted, sigma=5)
    lower_cp_smooth = np.maximum(0, gaussian_filter1d(lower_cp_sorted, sigma=5))
    upper_cp_smooth = gaussian_filter1d(upper_cp_sorted, sigma=5)

    plt.figure(figsize=(11, 6))
    plt.scatter(X_test, Y_test, color='gray', alpha=0.3, s=10, label='Реальные сделки')
    plt.plot(x_sorted, y_pred_smooth, color='black', linewidth=2, label='Предсказание (Ср. цена)')
    
    plt.fill_between(x_sorted, lower_naive_smooth, upper_naive_smooth, 
                     color='red', alpha=0.2, label='Наивный интервал (усечен по 0)')
    
    plt.plot(x_sorted, upper_cp_smooth, color='#2980b9', linestyle='-', linewidth=2.5, label='Conformal 95% (Адаптивный)')
    plt.plot(x_sorted, lower_cp_smooth, color='#2980b9', linestyle='-', linewidth=2.5)

    plt.title('Гарантии Conformal Prediction (Сглаженные границы прайсинга)')
    plt.xlabel('Медианный доход в районе')
    plt.ylabel('Стоимость недвижимости')
    plt.xlim(0, 10)
    plt.ylim(-0.5, 7) 
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('02_conformal_comparison.png', dpi=300)
    plt.close()

# =========================================================
# КЕЙС 3: Калибровка (Изотоническая регрессия vs Платт)
# =========================================================
def plot_comprehensive_calibration():
    print("--- ЗАПУСК КЕЙСА 3: КАЛИБРОВКА (Реальный антифрод) ---")
    
    try:
        data = fetch_openml(name='creditcard', version=1, parser='auto', as_frame=False)
        X = data.data.astype(np.float32)
        y = data.target.astype(int)
        
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=50000, random_state=42)
        for _, sample_index in sss.split(X, y):
            X, y = X[sample_index], y[sample_index]
            
    except Exception as e:
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=50000, n_features=30, weights=[0.97, 0.03], random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)
    X_calib, X_val, y_calib, y_val = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample', max_depth=18, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    probs_uncal_val = rf.predict_proba(X_val)[:, 1]

    try:
        from sklearn.frozen import FrozenEstimator
        iso = CalibratedClassifierCV(estimator=FrozenEstimator(rf), method='isotonic')
        platt = CalibratedClassifierCV(estimator=FrozenEstimator(rf), method='sigmoid')
    except ImportError:
        iso = CalibratedClassifierCV(estimator=rf, method='isotonic', cv='prefit')
        platt = CalibratedClassifierCV(estimator=rf, method='sigmoid', cv='prefit')

    iso.fit(X_calib, y_calib)
    probs_iso = iso.predict_proba(X_val)[:, 1]
    
    platt.fit(X_calib, y_calib)
    probs_platt = platt.predict_proba(X_val)[:, 1]

    brier_iso = brier_score_loss(y_val, probs_iso)
    brier_platt = brier_score_loss(y_val, probs_platt)
    unique_scores_iso = len(np.unique(probs_iso))
    unique_scores_platt = len(np.unique(probs_platt))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    fop_iso, mpv_iso = calibration_curve(y_val, probs_iso, n_bins=8, strategy='quantile')
    fop_platt, mpv_platt = calibration_curve(y_val, probs_platt, n_bins=16, strategy='quantile')

    ax1.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=2, label='Идеальная калибровка (y=x)')
    ax1.plot(mpv_platt, fop_platt, 'x-', color='#e74c3c', linewidth=2, markersize=8, label=f'Изотоническая ')
    ax1.plot(mpv_iso, fop_iso, 'o-', color='#27ae60', linewidth=2, markersize=6, label=f'Метод Платта ')
    
    max_val = max(0, np.max(mpv_platt)) 
    ax1.set_xlim([0, max_val])
    ax1.set_ylim([0, max_val])
    ax1.set_title('1. Диаграмма калибровки (Реальный антифрод)')
    ax1.set_xlabel('Предсказанная вероятность фрода')
    ax1.set_ylabel('Фактическая доля фрода')
    ax1.legend(loc='upper left')

    sort_idx = np.argsort(probs_uncal_val)
    raw_sorted = probs_uncal_val[sort_idx]
    iso_sorted = probs_iso[sort_idx]
    platt_sorted = probs_platt[sort_idx]

    ax2.plot(raw_sorted, iso_sorted, color='#e74c3c', linewidth=3, alpha=0.8,
             label=f'Изотоническая (Уникальных: {unique_scores_iso})')
    ax2.plot(raw_sorted, platt_sorted, color='#27ae60', linewidth=3,
             label=f'Platt Scaling (Уникальных: {unique_scores_platt})')
    
    ax2.set_xlim([min(raw_sorted) - 0.05, max(raw_sorted) + 0.05])
    ax2.set_title('2. Форма преобразования баллов в вероятность')
    ax2.set_xlabel('Сырой выход (Score) случайного леса')
    ax2.set_ylabel('Откалиброванная вероятность')
    ax2.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('03_calibration_imbalance.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    # plot_dml_comparison()
    # plot_conformal_heteroskedasticity()
    plot_comprehensive_calibration()
    print("Все расчеты и визуализации успешно обновлены!")