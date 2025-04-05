"import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Veri setini yükleyelim (Iris veri seti )
from sklearn.datasets import load_iris
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='species')

# Veri setini inceleme
print("Veri seti boyutu:", X.shape)
print("\nİlk 5 satır:")
print(X.head())
print("\nÖznitelik istatistikleri:")
print(X.describe())

# Hedef değişkeninin dağılımını inceleyelim
plt.figure(figsize=(8, 5))
y.value_counts().plot(kind='bar')
plt.title('Sınıf Dağılımı')
plt.xlabel('Sınıf')
plt.ylabel('Frekans')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('sinif_dagilimi.png')
plt.show()

# Öznitelikler arasındaki ilişkileri görselleştirelim
plt.figure(figsize=(12, 10))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Öznitelikler Arasındaki Korelasyon')
plt.tight_layout()
plt.savefig('korelasyon_matrisi.png')
plt.show()

# Özniteliklerin dağılımını inceleyelim
plt.figure(figsize=(15, 10))
for i, column in enumerate(X.columns):
    plt.subplot(2, 2, i+1)
    sns.histplot(data=X, x=column, hue=y, kde=True)
    plt.title(f'{column} Dağılımı')
plt.tight_layout()
plt.savefig('oznitelik_dagilimi.png')
plt.show()

# Veriyi eğitim ve test olarak bölelim - test setini daha büyük tutalım (%30 yerine %40)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

# Veriyi ölçeklendirelim
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sınıflandırma modellerini tanımlayalım
models = {
    'KNN': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'Naive Bayes': GaussianNB()
}

# Her model için en iyi parametreleri bulalım
best_params = {}
best_models = {}

# Çapraz doğrulama için StratifiedKFold kullanarak daha güvenilir sonuçlar elde edelim
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# KNN için en iyi k değerini bulalım
k_range = list(range(1, 31))
knn_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    knn_scores.append(scores.mean())

plt.figure(figsize=(10, 6))
plt.plot(k_range, knn_scores)
plt.xlabel('K Değeri')
plt.ylabel('Çapraz Doğrulama Doğruluk Skoru')
plt.title('KNN: K Değerine Karşı Doğruluk')
plt.grid(True)
plt.savefig('knn_k_degeri.png')
plt.show()

# Yüksek k değerlerine daha fazla ağırlık verelim, overfitting'i azaltmak için
# k değeri çok küçük olduğunda overfitting riski daha yüksektir
best_k = k_range[knn_scores.index(max(knn_scores))]
if best_k < 5:  # Eğer best_k çok küçükse, biraz daha yüksek bir değer seçelim
    best_k_candidates = [k for k in range(5, 15)]
    best_k_scores = [knn_scores[k-1] for k in best_k_candidates]
    if max(best_k_scores) >= max(knn_scores) * 0.95:  # Eğer performans çok düşmüyorsa
        best_k = best_k_candidates[best_k_scores.index(max(best_k_scores))]

best_params['KNN'] = {'n_neighbors': best_k}
best_models['KNN'] = KNeighborsClassifier(n_neighbors=best_k)

# Diğer modeller için parametre araması - overfitting'i azaltmak için parametreleri ayarlayalım
param_grids = {
    'Logistic Regression': {'C': [0.001, 0.01, 0.1, 1, 10, 100]},
    'Decision Tree': {
        'max_depth': [3, 5, 7, 10],  # Sınırlı derinlik (None yerine) overfitting'i azaltır
        'min_samples_split': [2, 5, 10], 
        'min_samples_leaf': [1, 2, 4]  # min_samples_leaf ekleyerek daha genelleştirilebilir ağaçlar oluşturalım
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],  # Daha fazla ağaç daha stabil sonuçlar verir
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
}

# Çapraz doğrulama kullanarak parametre optimizasyonu yapalım
for model_name, param_grid in param_grids.items():
    grid_search = GridSearchCV(models[model_name], param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)
    best_params[model_name] = grid_search.best_params_
    best_models[model_name] = grid_search.best_estimator_

# Naive Bayes için parametre araması yok
best_models['Naive Bayes'] = GaussianNB()

# Model değerlendirme ve tahmin performansı için çapraz doğrulama kullanarak daha güvenilir sonuçlar alalım
cv_results = {
    'Model': [],
    'CV_Accuracy': [],
    'CV_Precision': [],
    'CV_Recall': [],
    'CV_F1': []
}

# Her model için çapraz doğrulama sonuçlarını kaydedelim
for model_name, model in best_models.items():
    cv_accuracy = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy').mean()
    cv_precision = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='precision_weighted').mean()
    cv_recall = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='recall_weighted').mean()
    cv_f1 = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1_weighted').mean()
    
    cv_results['Model'].append(model_name)
    cv_results['CV_Accuracy'].append(cv_accuracy)
    cv_results['CV_Precision'].append(cv_precision)
    cv_results['CV_Recall'].append(cv_recall)
    cv_results['CV_F1'].append(cv_f1)

# Çapraz doğrulama sonuçlarını bir DataFrame'e dönüştürelim
cv_results_df = pd.DataFrame(cv_results)
cv_results_df = cv_results_df.sort_values('CV_Accuracy', ascending=False).reset_index(drop=True)

# Çapraz doğrulama sonuçlarını gösterelim
print("\nModellerin Çapraz Doğrulama Performans Karşılaştırması:")
print(cv_results_df)

# Sonuçları saklamak için bir sözlük oluşturalım
test_results = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': []
}

# Her modeli eğitelim ve test veri seti üzerinde değerlendirelim
for model_name, model in best_models.items():
    # Modeli eğitelim
    model.fit(X_train_scaled, y_train)
    
    # Test seti üzerinde tahmin yapalım
    y_pred = model.predict(X_test_scaled)
    
    # Metrikleri hesaplayalım
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Sonuçları kaydedelim
    test_results['Model'].append(model_name)
    test_results['Accuracy'].append(accuracy)
    test_results['Precision'].append(precision)
    test_results['Recall'].append(recall)
    test_results['F1 Score'].append(f1)
    
    # Karmaşıklık matrisini görselleştirelim
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
    plt.title(f'{model_name} Karmaşıklık Matrisi')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.ylabel('Gerçek Sınıf')
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.show()
    
    # ROC eğrisini çizelim (multinominal sınıflandırma için çoklu ROC eğrileri)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test_scaled)
        n_classes = len(np.unique(y))
        
        plt.figure(figsize=(8, 6))
        for i in range(n_classes):
            y_test_bin = np.where(y_test == i, 1, 0)
            y_score_class = y_score[:, i]
            
            fpr, tpr, _ = roc_curve(y_test_bin, y_score_class)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, label=f'Sınıf {data.target_names[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Yanlış Pozitif Oranı')
        plt.ylabel('Doğru Pozitif Oranı')
        plt.title(f'{model_name} - ROC Eğrisi')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(f'{model_name.lower().replace(" ", "_")}_roc_curve.png')
        plt.show()
    
    # Sınıflandırma raporunu yazdıralım
    print(f"\n{model_name} Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))

# Test sonuçlarını bir DataFrame'e dönüştürelim
test_results_df = pd.DataFrame(test_results)
test_results_df = test_results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)

# Test sonuçlarını gösterelim
print("\nTest Veri Seti Üzerinde Modellerin Performans Karşılaştırması:")
print(test_results_df)

# Çapraz doğrulama ve test sonuçlarını karşılaştıralım
comparison_df = pd.merge(cv_results_df, test_results_df, on='Model')
comparison_df['Accuracy_Diff'] = comparison_df['Accuracy'] - comparison_df['CV_Accuracy']
print("\nÇapraz Doğrulama ve Test Sonuçları Karşılaştırması (Accuracy_Diff büyükse overfitting olabilir):")
print(comparison_df[['Model', 'CV_Accuracy', 'Accuracy', 'Accuracy_Diff']])

# Modellerin performansını görselleştirelim
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

plt.figure(figsize=(14, 10))
for i, metric in enumerate(metrics):
    plt.subplot(2, 2, i+1)
    sns.barplot(x='Model', y=metric, data=test_results_df)
    plt.title(f'Model Karşılaştırması: {metric}')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')

plt.tight_layout()
plt.savefig('model_karsilastirmasi.png')
plt.show()

# En iyi performans gösteren modeli gösterelim
best_model_name = cv_results_df.iloc[0]['Model']  # Çapraz doğrulama sonuçlarına göre en iyi model
best_model = best_models[best_model_name]
print(f"\nEn iyi performans gösteren model (Çapraz Doğrulama sonuçlarına göre): {best_model_name}")
print(f"En iyi parametreler: {best_params.get(best_model_name, 'Parametre araması yapılmadı')}")
print(f"Çapraz Doğrulama doğruluğu: {cv_results_df.iloc[0]['CV_Accuracy']:.4f}")
print(f"Test seti doğruluğu: {test_results_df[test_results_df['Model'] == best_model_name]['Accuracy'].values[0]:.4f}")

# Öğrenme eğrilerini çizelim
plt.figure(figsize=(12, 8))
for model_name, model in best_models.items():
    train_sizes = np.linspace(0.1, 1.0, 10)
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train_scaled, y_train, train_sizes=train_sizes, cv=cv, scoring='accuracy'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.plot(train_sizes, train_mean, 'o-', color=f'C{list(best_models.keys()).index(model_name)}',
                 label=f'{model_name} (Eğitim)')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1,
                         color=f'C{list(best_models.keys()).index(model_name)}')
        
        plt.plot(train_sizes, test_mean, 's-', color=f'C{list(best_models.keys()).index(model_name)}',
                 label=f'{model_name} (Doğrulama)')
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1,
                         color=f'C{list(best_models.keys()).index(model_name)}')
    except Exception as e:
        print(f"Uyarı: {model_name} için öğrenme eğrisi çizilirken bir hata oluştu. Hata: {e}")

plt.xlabel('Eğitim Örnek Sayısı')
plt.ylabel('Doğruluk')
plt.title('Öğrenme Eğrileri')
plt.grid(True)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('ogrenme_egrileri.png')
plt.show()
