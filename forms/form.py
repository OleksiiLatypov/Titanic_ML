from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, FloatField, SelectField
from wtforms.validators import InputRequired, NumberRange


class PassengerForm(FlaskForm):
    pclass = IntegerField('Passenger Class', validators=[InputRequired(), NumberRange(min=1, max=3)])
    sex = SelectField(choices=[('male', 'Male'), ('female', 'Female')], validators=[InputRequired()])
    age = FloatField('Age', validators=[InputRequired(), NumberRange(min=0.0, max=100.0)])
    name = StringField('Name', validators=[InputRequired()])
    siblings_spouses = IntegerField('Siblings/Spouses Aboard', validators=[InputRequired(), NumberRange(min=0, max=100)])
    parents_children = IntegerField('Parents/Children Aboard', validators=[InputRequired(), NumberRange(min=0, max=100)])
    fare = FloatField('Fare', validators=[InputRequired(), NumberRange(min=0, max=300)])
    embarked = SelectField('Embarked', choices=[('C', 'Cherbourg'), ('Q', 'Queenstown'), ('S', 'Southampton')],
                           validators=[InputRequired()])
