from flask import Flask, render_template, session, redirect, url_for
from flask.ext.wtf import Form
from wtforms import IntegerField, StringField, SubmitField, SelectField, DecimalField, TextAreaField
from wtforms.validators import Required
import pickle


from transformers import *

reload(sys)  
sys.setdefaultencoding('utf8')
# Initialize Flask App
app = Flask(__name__)


print "loading my model"
with open('final_model.pkl', 'rb') as handle:
    machine_learning_model = pickle.load(handle)
print "model loaded"


# Initialize Form Class
class theForm(Form):
    param1 = TextAreaField(label='Script:',  validators=[Required()])
    param2 = StringField(label='Genre:', validators=[Required()])
    param3 = IntegerField(label='Number of Pages:', validators=[Required()])
    submit = SubmitField('Submit')


@app.route('/', methods=['GET', 'POST'])
def home():
    print session
    form = theForm(csrf_enabled=False)
    if form.validate_on_submit():  # activates this if when i hit submit!
        # Retrieve values from form
        session['script'] = form.param1.data
        session['genre'] = form.param2.data
        session['runtime'] = form.param3.data
       
        # Create array from values
        data_frame = pd.DataFrame({'script': [session['script']], 'Runtime': [session['runtime']], 'Genre': [session['genre']] })
        
        

        # Return only the Predicted iris species
        session['prediction'] = '%.2f'%(float(machine_learning_model.predict(data_frame)))
        figure = PlotSentiment(session['script'],session['runtime'])
        session['y']= figure.get_plot()[1]
        session['x']= figure.get_plot()[0]
        session['text']= figure.get_plot()[2]        
        # Implement Post/Redirect/Get Pattern
        return redirect(url_for('home'))

    return render_template('home.html', form=form, **session)


# Handle Bad Requests
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

app.secret_key = 'super_secret_key_shhhhhh'
if __name__ == '__main__':
    app.run(debug=True)
