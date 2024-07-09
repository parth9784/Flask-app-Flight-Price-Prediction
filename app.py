from flask import Flask,render_template,request
import numpy as np
import joblib
app=Flask(__name__)

@app.route("/")
def wel():
    return render_template("index.html")
@app.route("/predict", methods=["GET","POST"])
def predict():
    if(request.method=="POST"):
        model=joblib.load("Flight_fare.pkl")
        airline=int(request.form["airline"])
        from_city=int(request.form["from"])
        to=int(request.form["to"])
        stop=int(request.form["stops"])
        class_=int(request.form["class"])
        dur=request.form["Duration"]
        day=request.form["left"]
        x=np.array([[airline,from_city,to,stop,class_,dur,day]])
        y_pred=model.predict(x)
        print(y_pred)
    return render_template("result.html",ans=y_pred)
if(__name__=="__main__"):
    app.run(debug=True)
