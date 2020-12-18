from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pygal
from flask import Flask, render_template, request
import warnings
warnings.filterwarnings("ignore")

dataSet = pd.read_csv('bank.csv', sep=';')

dataSet = dataSet.drop(dataSet[dataSet.poutcome == 'other'].index)
dataSet = dataSet.drop(dataSet[dataSet.education == 'unknown'].index) 
dataSet = dataSet.drop(dataSet[dataSet.job == 'unknown'].index)
dataSet = dataSet.drop(['day'], axis=1)
dataSet.rename(index=str, columns={'y': 'result'}, inplace=True)

dataSet['housing'] = dataSet['housing'].map({'yes': 1, 'no': 0})
dataSet['default'] = dataSet['default'].map({'yes': 1, 'no': 0})
dataSet['loan'] = dataSet['loan'].map({'yes': 1, 'no': 0})
dataSet['result'] = dataSet['result'].map({'yes': 1, 'no': 0})

dataSet1 = dataSet.iloc[:, 0:-1]
str_features = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']

for feature in str_features:
    dataSet1 = pd.get_dummies(dataSet1, columns = [feature])

data_result = pd.DataFrame(dataSet['result'])
dataSet1 = pd.merge(dataSet1, data_result, left_index = True, right_index = True)
db = dataSet1.values
columns = dataSet1.loc[:, dataSet1.columns != 'result'].columns
X = db[:,0:-1]
y = db[:,-1]

sm = SMOTE(sampling_strategy='auto', k_neighbors=8, random_state=8)
X_res, y_res = sm.fit_resample(X, y)
X_res = np.concatenate((X, X_res))
y_res = np.concatenate((y, y_res))

X_res = pd.DataFrame(data=X_res, columns=columns)
y_res = pd.DataFrame(data=y_res, columns=['result'])

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.1, random_state=8)
model = LogisticRegression()
model = model.fit(X_train, y_train)
predictions = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
cm = confusion_matrix(y_test, predictions)
#plot_confusion_matrix(model, X_test, y_test, values_format='', cmap='Oranges')
# plt.show()

app = Flask(__name__)
posts =[
    {
        'author':'Tùng Dương',
        'title':'Dự Đoán',
        'content':'Dự đoán phản hồi khách hàng mới từ 10/05',
        'date_posted':'20/05, 2020'
    },
    {
        'author':'Việt Hà',
        'title':'Thống kê',
        'content':'Xu hướng của khách hàng từ 10/04 đến 10/05',
        'date_posted':'15/05, 2020'
    },
    {
        'author':'Phạm Dũng',
        'title':'Doanh Thu',
        'content':'Doanh thu quý I năm 2020',
        'date_posted':'01/05, 2020'

    },
    {
        'author':'Dinh Phan',
        'title':'Ưu đãi',
        'content':'Đề xuất những ưu đãi cho từng cụm khách hàng',
        'date_posted':'24/04, 2020'
    }
]

@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html", posts=posts)

display = dataSet[dataSet['result'] == 1]
display = display.drop(['poutcome'], axis=1)
display = display.drop(['balance'], axis=1)
display = display.drop(['contact'], axis=1)
display = display.drop(['default'], axis=1)
display = display.loc[:, display.columns != 'result']

@app.route("/predict")
def predict():
	return render_template('predict.html', tables=[display.to_html(classes='data')],
		title='Dự đoán', titles=display.columns.values)

@app.route("/chart/response")
def response():
    p_lb0 = sum(cm[:,0]) / sum(sum(cm)) * 100
    p_lb1 = 100 - p_lb0
    lb0 = sum(cm[:,0])
    lb1 = sum(cm[:,1])
    chart = pygal.Pie()
    chart.title="Phản hồi của khách hàng"
    chart.add("không đăng ký", lb0)
    chart.add("có đăng ký", lb1)
    graphdata = chart.render_data_uri()
    return render_template("response.html", graph_data=graphdata, title='Thống kê phản hồi', lb0=lb0, lb1=lb1)

lst = [dataSet]
for column in lst:
    column.loc[column["age"] < 30,  'age_group'] = 20
    column.loc[(column["age"] >= 30) & (column["age"] <= 39), 'age_group'] = 30
    column.loc[(column["age"] >= 40) & (column["age"] <= 49), 'age_group'] = 40
    column.loc[(column["age"] >= 50) & (column["age"] <= 59), 'age_group'] = 50
    column.loc[column["age"] >= 60, 'age_group'] = 60

count_age_response = pd.crosstab(dataSet['result'], dataSet['age_group']).apply(lambda x: x/x.sum() * 100)
count_age_response = count_age_response.transpose() 
count_age_response.to_numpy()
str_lab = ["<30", '30-39', '40-49', '50-59', '>=60']

@app.route("/chart/agegroup")
def agegroup():
	chart = pygal.Bar()
	chart.title = "Phản hồi của khách hàng theo nhóm tuổi"
	chart.x_labels = str_lab
	chart.add("Không đăng ký", count_age_response[0].round(2))
	chart.add("Có đăng ký", count_age_response[1].round(2))
	graphdata = chart.render_data_uri()
	return render_template("agegroup.html", graph_data=graphdata, title='Phản hồi theo nhóm tuổi')


@app.route("/chart/job")
def job():
	chart = pygal.Bar()
	data = dataSet[dataSet['result'] == 1]
	jobs = ['blue-collar', 'entrepreneur', 'services', 'unemployed', 'technician', 'self-employed', 
	'admin.', 'housemaid', 'management','student', 'retired']
	numbers = data.shape[0]
	for job in jobs:
		data_job = data[data['job'] == job]
		ratio = round(100 * data_job.shape[0] / numbers, 2)
		chart.add(job, ratio)
	chart.title = "Phản hồi của khách hàng theo nghề nghiệp"
	graphdata = chart.render_data_uri()
	return render_template("job.html", graph_data=graphdata, title='Phản hồi theo nghề nghiệp')

@app.route("/chart/month")
def month():
    chart = pygal.HorizontalBar()
    data = dataSet[dataSet['result'] == 1]
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep','oct', 'nov', 'dec']
    numbers = data.shape[0]
    for month in months:
        data_month = data[data['month'] == month]
        ratio = round(100 * data_month.shape[0] / numbers, 2)
        chart.add(month, ratio)
    chart.title = "Phản hồi của khách hàng vào các tháng trong năm"
    graphdata = chart.render_data_uri()
    return render_template("month.html", graph_data=graphdata, title='Phản hồi trong tháng')

@app.route("/chart/default")
def default():
    yes_res = dataSet[dataSet['result'] == 1]
    yes_res_no_default = yes_res[yes_res['default'] == 0]
    yes_res_yes_default = yes_res[yes_res['default'] == 1]
    chart = pygal.Pie()
    chart.title = "Vấn đề nợ xấu"
    chart.add("Không có nợ xấu", len(yes_res_no_default))
    chart.add("Có nợ xấu", len(yes_res_yes_default))
    graphdata = chart.render_data_uri()
    return render_template("default.html", graph_data=graphdata, title='Vấn đề nợ xấu')

@app.route("/upload", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
    	print(request.files['file'])
    	f = request.files['file']
    	data_xls_org = pd.read_excel(f)
    	data_xls = pd.read_excel(f)

    	data_xls = data_xls.drop(['day'], axis=1)
    	data_xls = data_xls.drop(data_xls[data_xls.poutcome == 'other'].index)
    	data_xls = data_xls.drop(data_xls[data_xls.education == 'unknown'].index)
    	data_xls = data_xls.drop(data_xls[data_xls.job == 'unknown'].index)

    	data_xls['housing'] = data_xls['housing'].map({'yes': 1, 'no': 0})
    	data_xls['default'] = data_xls['default'].map({'yes': 1, 'no': 0})
    	data_xls['loan'] = data_xls['loan'].map({'yes': 1, 'no': 0})

    	features = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']

    	for feature in features:
    		data_xls = pd.get_dummies(data_xls, columns = [feature])

    	sub_proba = model.predict_proba(data_xls)
    	sub_proba = pd.DataFrame(sub_proba[:,1])
    	sub_proba = sub_proba.apply(lambda x: round(100*x, 2))

    	data_xls_org = pd.merge(data_xls_org, sub_proba, left_index = True, right_index = True)
    	data_xls_org.rename(index=str, columns={0: 'result(%)'}, inplace=True)

    	data_xls_org = data_xls_org.drop(['default'], axis=1)
    	data_xls_org = data_xls_org.drop(['balance'], axis=1)
    	data_xls_org = data_xls_org.drop(['previous'], axis=1)
    	data_xls_org = data_xls_org.drop(['poutcome'], axis=1)
    	data_xls_org = data_xls_org.drop(['contact'], axis=1)

    	return render_template('upload.html', tables=[data_xls_org.to_html(classes='data')], 
        	titles=data_xls_org.columns.values, title='Upload')
    return render_template("upload0.html", title='Upload')

if __name__ == '__main__':
    app.run(debug=True)