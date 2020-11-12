
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


training_dataset = pd.read_csv('Training.csv')
test_dataset = pd.read_csv('Testing.csv')


X = training_dataset.iloc[:, 0:132].values
y = training_dataset.iloc[:, -1].values


dimensionality_reduction = training_dataset.groupby(training_dataset['prognosis']).max()
diseases = dimensionality_reduction.index
diseases = pd.DataFrame(diseases)



from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth = 28)
classifier.fit(X_train, y_train)

st.title("DISEASE PREDICTION")

def user():
    
    user_input = []
    count = 0
    
    if st.checkbox("itching"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1

    if st.checkbox("skin_rash"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1

    if st.checkbox("nodal_skin_eruptions"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1

    if st.checkbox("continuous_sneezing"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1

    if st.checkbox("shivering"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
    if st.checkbox("chills"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("joint_pain"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1        

    if st.checkbox("stomach_pain"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
        
    if st.checkbox("acidity"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("ulcers_on_tongue"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("muscle_wasting"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("vomiting"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("burning_micturition"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("spotting_ urination"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("fatigue"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("weight_gain"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("anxiety"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("cold_hands_and_feets"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("mood_swings"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("weight_loss"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("restlessness"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("lethargy"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("patches_in_throat"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("irregular_sugar_level"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
        
    if st.checkbox("cough"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
    
    if st.checkbox("high_fever"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("sunken_eyes"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("breathlessness"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("sweating"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("dehydration"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("indigestion"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("headache"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("yellowish_skin"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("dark_urine"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("nausea"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("loss_of_appetite"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("pain_behind_the_eyes"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("back_pain"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("constipation"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("abdominal_pain"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("diarrhoea"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("mild_fever"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("yellow_urine"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("yellowing_of_eyes"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("acute_liver_failure"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("fluid_overload"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("swelling_of_stomach"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("swelled_lymph_nodes"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("malaise"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("blurred_and_distorted_vision"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("phlegm"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("throat_irritation"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("redness_of_eyes"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("sinus_pressure"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("runny_nose"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("congestion"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("chest_pain"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("weakness_in_limbs"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
    
    if st.checkbox("fast_heart_rate"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("pain_during_bowel_movements"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("pain_in_anal_region"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("bloody_stool"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("irritation_in_anus"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("neck_pain"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("dizziness"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("cramps"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("bruising"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("obesity"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("swollen_legs"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("swollen_blood_vessels"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("puffy_face_and_eyes"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("enlarged_thyroid"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("brittle_nails"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("swollen_extremeties"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("excessive_hunger"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("extra_marital_contacts"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("drying_and_tingling_lips"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("slurred_speech"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("knee_pain"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("hip_joint_pain"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("muscle_weakness"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("stiff_neck"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("swelling_joints"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("movement_stiffness"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("spinning_movements"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("loss_of_balance"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("unsteadiness"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("weakness_of_one_body_side"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("loss_of_smell"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("bladder_discomfort"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("foul_smell_of urine"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("continuous_feel_of_urine"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("passage_of_gases"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("internal_itching"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("toxic_look_(typhos)"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("depression"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("irritability"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("muscle_pain"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
    
    if st.checkbox("altered_sensorium"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("red_spots_over_body"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("belly_pain"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("abnormal_menstruation"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("dischromic _patches"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("watering_from_eyes"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("increased_appetite"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("polyuriafamily_history"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("mucoid_sputum"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("rusty_sputum"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("lack_of_concentration"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("visual_disturbances"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("receiving_blood_transfusion"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("receiving_unsterile_injections"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("coma"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("stomach_bleeding"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("distention_of_abdomen"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("history_of_alcohol_consumptionfluid_overload"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
    
    if st.checkbox("blood_in_sputum"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("prominent_veins_on_calf"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("palpitations"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("painful_walking"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("pus_filled_pimples"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("blackheads"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("scurring"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("skin_peeling"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("silver_like_dustingsilver_like_dusting"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("small_dents_in_nails"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("inflammatory_nails"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("blister"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("red_sore_around_nose"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
        
    if st.checkbox("yellow_crust_ooze"):
        user_input.append(1)
        count += 1
    else:
        user_input.append(0)
        count += 1
    
    temp = [0] * (132-count)

    user_input= user_input +temp

    
    return user_input
# y_pred will give the predicted value
    
doc_dataset = pd.read_csv('D:\Project\doctors_dataset.csv', names = ['Name', 'Description'])


diseases = dimensionality_reduction.index
diseases = pd.DataFrame(diseases)

doctors = pd.DataFrame()
doctors['name'] = np.nan
doctors['link'] = np.nan
doctors['disease'] = np.nan

doctors['disease'] = diseases['prognosis']


doctors['name'] = doc_dataset['Name']
doctors['link'] = doc_dataset['Description']




inp = user()

if st.button("Predict"):
    if 1 not in inp:
        st.subheader("*No symptoms detected*")
        st.subheader("*Please go back and select the symptoms you have**")
    else:
        y_pred = int(classifier.predict([inp]))
        
        ans = list(labelencoder.inverse_transform([y_pred]))
        st.header("You might have: " + ans[0])
        
        row = doctors[doctors['disease'] == ans[0]]
        n = row['name'].values
        st.header('We suggest you to consult: '+ n[0])
        l = row['link'].values
        st.subheader('To book an appointment Visit:-')
        st.subheader(l[0])
                    










































