from flask import Flask, render_template, session, redirect, url_for
from flask_bootstrap import Bootstrap
from flask.ext.wtf import Form
from wtforms import IntegerField, StringField, SubmitField, SelectField, DecimalField
from wtforms.validators import Required
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import simplejson
import sys
import logging
import pickle

# Load Iris Data
iris_data = load_iris()
features = iris_data.data
feature_names = iris_data.feature_names
target = iris_data.target
target_names = iris_data.target_names

# Initialize Flask App
app = Flask(__name__)


# Initialize Form Class
class theForm(Form):
    sepal_length = DecimalField('Sepal Length (cm):', places=2, validators=[Required()])
    sepal_width = DecimalField('Sepal Width (cm):', places=2, validators=[Required()])
    petal_length = DecimalField('Petal Length (cm):', places=2, validators=[Required()])
    petal_width = DecimalField('Petal Width (cm):', places=2, validators=[Required()])
    submit = SubmitField('Submit')


@app.route('/', methods=['GET', 'POST'])
def model():
    form = theForm(csrf_enabled=False)
    if form.validate_on_submit():  # activates this if when i hit submit!
        # Retrieve values from form
        session['sepal_length'] = form.sepal_length.data
        session['sepal_width'] = form.sepal_width.data
        session['petal_length'] = form.petal_length.data
        session['petal_width'] = form.petal_width.data
        # Create array from values
        flower_instance = [(session['sepal_length']), (session['sepal_width']), (session['petal_length']),
                           (session['petal_width'])]
        # Fit model with n_neigh neighbors

        with open('knn.pickle', 'rb') as handle:
            knn = pickle.load(handle)
            session['params'] = knn.get_params()
        # Return only the Predicted iris species
        session['prediction'] = target_names[knn.predict(flower_instance)][0].capitalize()
        # Implement Post/Redirect/Get Pattern
        return redirect(url_for('model'))

    return render_template('model.html', form=form, **session)


# Handle Bad Requests
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


app.secret_key = 'super_secret_key'

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

if __name__ == '__main__':
    app.run(debug=True)
