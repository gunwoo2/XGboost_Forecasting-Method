import json
import pandas as pd
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np

# 파일 경로를 지정.
file_path = "본인의 Path"

# TXT 파일에서 JSON 데이터를 읽기
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 데이터 추출 및 전처리
rows = []
for key, item in data.items():
    row = {
        "SvYear": item.get("SvYear"),
        "Werks": item.get("Werks"),
        "Matnr": item.get("Matnr"),
        "Savqu": item.get("Savqu"),
        "Meins": item.get("Meins"),
        "Totsal": item.get("Totsal"),
        "Netpr": item.get("Netpr"),
        "Waers": item.get("Waers")
    }
    for i in range(1, 13):
        row[f"Svqty{i}"] = item.get(f"Svqty{i}")
    rows.append(row)

# 데이터프레임 생성
df = pd.DataFrame(rows)

# 숫자형 데이터 변환
numeric_columns = ['SvYear', 'Savqu', 'Totsal', 'Netpr'] + [f'Svqty{i}' for i in range(1, 13)]
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# One-Hot Encoding을 사용하여 문자열 데이터 처리
categorical_columns = ['Werks', 'Matnr', 'Meins', 'Waers']
df = pd.get_dummies(df, columns=categorical_columns, dummy_na=True)

# NaN 값 0으로 대체
df.fillna(0, inplace=True)

# X와 y 설정
y_columns = [f'Svqty{i}' for i in range(1, 13)]
X = df.drop(columns=y_columns)
y = df[y_columns]

# 데이터 분할을 하지 않고 전체 데이터에 대해 모델 학습
model = xgb.XGBRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 예측 수행
y_pred = model.predict(X)

# 결과를 JSON 형식으로 저장
output_data = {}
for index, row in X.reset_index(drop=True).iterrows():
    werks_col = row.filter(like='Werks_').idxmax(axis=0).replace('Werks_', '')
    matnr_col = row.filter(like='Matnr_').idxmax(axis=0).replace('Matnr_', '')
    meins_col = row.filter(like='Meins_').idxmax(axis=0).replace('Meins_', '')
    waers_col = row.filter(like='Waers_').idxmax(axis=0).replace('Waers_', '')
    
    output_data[index] = {
        "SvYear": int(row['SvYear']),
        "Werks": werks_col,
        "Matnr": matnr_col,
        "Savqu": int(row['Savqu']),
        "Meins": meins_col,
        "Totsal": float(row['Totsal']),
        "Netpr": float(row['Netpr']),
        "Waers": waers_col,
        "Predictions": [float(pred) for pred in y_pred[index]]
    }

# 결과를 JSON 파일로 저장
output_file_path = "output_json_data.json"
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    json.dump(output_data, output_file, ensure_ascii=False, indent=4)

print("JSON 파일이 성공적으로 생성되었습니다.")
##########################################################################
##########################################################################

# JSON 파일 경로
json_file_path = "output_json_data.json"

# JSON 파일에서 예측 결과 데이터 불러오기
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    predictions_data = json.load(json_file)

# 플랜트와 자재별 월별 판매량을 저장할 딕셔너리 초기화
plant_material_monthly_sales = {}

# 예측 데이터에서 플랜트와 자재별 월별 판매량 추출
for index, data in predictions_data.items():
    plant = data['Werks']
    material = data['Matnr']
    predictions = data['Predictions']
    
    # 플랜트-자재별 월별 판매량
    if (plant, material) not in plant_material_monthly_sales:
        plant_material_monthly_sales[(plant, material)] = {month: [] for month in range(1, 13)}
    for month, prediction in enumerate(predictions, start=1):
        plant_material_monthly_sales[(plant, material)][month].append(prediction)

# 플랜트와 자재별 월별 평균 판매량 계산
average_sales_by_plant_material_month = {}
for (plant, material), sales_data in plant_material_monthly_sales.items():
    monthly_sales = [sum(sales) / len(sales) for sales in sales_data.values()]
    average_sales_by_plant_material_month[(plant, material)] = monthly_sales

# 플랜트와 자재 정보 추출
plants = sorted(set(plant for plant, _ in average_sales_by_plant_material_month.keys()))
materials = sorted(set(material for _, material in average_sales_by_plant_material_month.keys()))

# 히트맵 데이터 생성
heatmap_data = pd.DataFrame(index=plants, columns=materials, dtype=float)
for (plant, material), monthly_sales in average_sales_by_plant_material_month.items():
    heatmap_data.at[plant, material] = sum(monthly_sales)

# 히트맵 그리기
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap='viridis', annot=True, fmt='.0f', cbar_kws={'label': 'Average Monthly Sales'})
plt.title('Average Monthly Sales by Plant and Material')
plt.xlabel('Material')
plt.ylabel('Plant')
plt.show()

##########################################################

# 각 플랜트별로 데이터를 분할하여 저장할 딕셔너리 초기화
plant_data = {}

# 각 플랜트별로 데이터를 그룹화하고 저장
for index, data in predictions_data.items():
    plant = data['Werks']
    if plant not in plant_data:
        plant_data[plant] = []
    plant_data[plant].append(data)

# 플랜트별로 분할된 데이터 확인
for plant, data_list in plant_data.items():
    print(f"Plant: {plant}")
    for data in data_list:
        print(data)
    print()

# 플랜트별로 산점도 행렬을 그리기 위한 함수 정의
def draw_scatter_matrix(plant_data):
    for plant, data_list in plant_data.items():
        print(f"Plant: {plant}")
        # 플랜트 내에서 자재별로 달별 판매량 추출
        material_sales = {}
        for data in data_list:
            material = data['Matnr']
            for month, sales in enumerate(data['Predictions'], start=1):
                if material not in material_sales:
                    material_sales[material] = {m: 0 for m in range(1, 13)}
                material_sales[material][month] += sales

        # 추출한 데이터를 DataFrame으로 변환
        df = pd.DataFrame([(plant, material, month, sales) 
                           for material, sales_data in material_sales.items() 
                           for month, sales in sales_data.items()],
                          columns=['Plant', 'Material', 'Month', 'Sales'])

        # 산점도 행렬 그리기
        sns.pairplot(df, hue='Material', vars=['Month', 'Sales'], diag_kind='kde', palette='viridis', height=3.5, aspect=1.5)
        plt.title(f"Plant {plant} - Material Sales Scatter Matrix")
        plt.show()

# 산점도 행렬 그리기
draw_scatter_matrix(plant_data)
