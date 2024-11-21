from flask import Flask,render_template,request
from src.pipeline.predict_pipeline import PredictPipeline,CustomData


app=Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="GET":
        return render_template("home.html")
    else:
        data=CustomData(
            CreditScore=request.form.get("CreditScore"),
            Age=request.form.get("Age"),
            Tenure=request.form.get("Tenure"),
            Balance=request.form.get("Balance"),
            NumOfProducts=request.form.get("NumOfProducts"),
            HasCrCard=request.form.get("HasCrCard"),
            IsActiveMember=request.form.get("IsActiveMember"),
            EstimatedSalary=request.form.get("EstimatedSalary"),
            Geography=request.form.get("Geography"),
            Gender=request.form.get("Gender"),
)
        
        
        pred_df=data.get_data_as_dataframe()
        print(pred_df)
        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template("home.html",results=results[0])



#   numerical_cols=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
    #    'IsActiveMember', 'EstimatedSalary']

        # categorical_cols=['Geography', 'Gender']
if __name__=="__main__":
    app.run("0.0.0.0",debug=True)
