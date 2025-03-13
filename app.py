from flask import Flask, render_template, request
import Outcome_predictor as op
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        fighter1 = request.form['fighter1']
        fighter2 = request.form['fighter2']

        outcome, pred1, pred2 = op.prdict_outcome(fighter1, fighter2)

        return render_template('index.html', outcome=outcome, pred1=pred1, pred2=pred2,
                               fighter1=fighter1, fighter2=fighter2, fighters=op.getFighterNames())
    else:
        return render_template('index.html', fighters=op.getFighterNames())

if __name__ == '__main__':
    app.run(debug=True)
