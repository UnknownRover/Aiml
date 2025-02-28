from flask import Flask, render_template, send_file

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/cluster')
def cluster():
    return send_file("cluster_distribution.png", mimetype='image/png')

@app.route('/confusion')
def confusion():
    return send_file("confusion_matrix.png", mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
