from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load("student_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = [[
        float(request.form["StudyHours"]),
        float(request.form["Attendance"]),
        float(request.form["PreviousMarks"]),
        float(request.form["SleepHours"]),
        float(request.form["AssignmentsCompleted"])
    ]]

    score = model.predict(data)[0]
    return render_template("result.html",
    score=round(score, 2))

if __name__ == "__main__":
    app.run(debug=True)