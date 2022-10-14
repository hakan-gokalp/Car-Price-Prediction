import streamlit as st 
import pandas as pd
import numpy as np
from PIL import Image
import carlist
import make_model_list
import lightgbm

import joblib


from Preprocessing import preprocess

filename = 'finalized_model.sav'
model = joblib.load(filename)


def main():
    st.title('Fenyx Car Price App')
    st.markdown("""
                :dart: This Streamlit App is made to predict vehicle price.""")
    st.markdown("<h3></h3>", unsafe_allow_html= True)
    image = Image.open('App.png')
    image_imp = Image.open('feature_importance.png')
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?", ("Online", "Batch"))
    st.sidebar.info('This app is created to predict Car Price')
    st.sidebar.image(image)
    st.sidebar.info('This app uses LGM Model')
    st.sidebar.image(image_imp)
    
    
    if add_selectbox == "Online":
       st.info("Input data below")
        #Based on our optimal features selection
             
       st.header("Features")
       st.subheader('Car Brand & Model')
       make= st.selectbox(f'Make',['Audi', 'BMW', 'Ford', 'Mercedes-Benz', 'Opel', 'Volkswagen', 'Renault', 'Citroen', 'Chevrolet', 'Dacia', 'Fiat', 'Honda', 'Hyundai', 'Kia', 'Mazda', 'Peugeot', 'Skoda', 'Toyota', 'Tesla', 'Volvo'])
       modell = st.selectbox(f'Model', carlist.model(make) )
       make_model = make+' '+modell
       make_model1= make_model_list.make_model_value(make_model)
       mileage = st.slider('Milage', min_value=0, max_value=500000 )
       age = st.slider('Vehicle Age', min_value=0, max_value=50)
       
       st.subheader('Fuel Type')
       gasoline = st.selectbox('Gasoline', ('No','Yes'))
       diesel = st.selectbox('Diesel', ('No','Yes'))
       lpg = st.selectbox('LPG', ('No','Yes'))
       electric = st.selectbox('Electric', ('No','Yes'))
       fuel_cons = st.slider('Fuel Consumption', min_value=0, max_value=20, value=10)
       def carbon_emission():
           if gasoline == 'Yes':
               return fuel_cons*24
           elif diesel == 'Yes':
               return fuel_cons*26.4
           elif lpg == 'Yes':
               return fuel_cons*83
           else:
               return 0
       
       
       st.subheader('Engine & Vehicle Weight')
       power = st.slider('Power (hp)', min_value=0, max_value=300, value = 90)
       engine_size = st.slider('Engine Size', min_value=0, max_value=3000, value= 1600)
       cylinders = st.slider('Number of Cylinders', min_value=1, max_value= 12, value=4)
       empty_weight = st.slider('Empty Weight', min_value=800, max_value= 4500, value=1200)
       general_inspection = st.selectbox('General Inspection', ('No', 'Yes'))
       
       
       st.subheader('Type')
       new = st.selectbox('New', ('No','Yes'))
       used = st.selectbox('Used', ('No','Yes'))
       
       
       st.subheader('Body Type')
       compact = st.selectbox('Compact', ('No','Yes'))
       sedan = st.selectbox('Sedan', ('No','Yes'))
       station = st.selectbox('Station Vagon', ('No','Yes'))
       convertable = st.selectbox('Convertable', ('No','Yes'))
       transporter = st.selectbox('Transporter', ('No','Yes'))
       off_road = st.selectbox('Off-Road/Pick-Up', ('No', 'Yes'))
       
             
              
       st.subheader('Transmission')
       automatic = st.selectbox('Automatic', ('No','Yes'))
       manual = st.selectbox('Manual', ('No','Yes'))
       semi_auto = st.selectbox('Semi Automatic', ('No','Yes'))
       
       st.subheader('Comfort & Convenience')
       air_cond = st.selectbox('Air Condition', ('No','Yes'))
       cruis = st.selectbox('Cruis Control', ('No','Yes'))
       mirrors = st.selectbox('Electric Side Mirrors', ('No','Yes'))
       door_lock = st.selectbox('Remote Door Lock', ('No','Yes'))
       windows = st.selectbox('Electric Windows', ('No','Yes'))
       seat_heat = st.selectbox('Seat Heating', ('No','Yes'))
       sunroof = st.selectbox('Sunroof', ('No','Yes'))
       pan_roof = st.selectbox('Panorama Roof', ('No', 'Yes'))
       wind_deflector = st.selectbox('Wind Deflector', ('No', 'Yes'))
       seat_ventilation = st.selectbox('Seat Ventilation', ('No', 'Yes'))
       
       st.subheader('Safety & Security')
       fog_lights = st.selectbox('Fog Lights', ('No','Yes'))
       immobilizier = st.selectbox('Immobilizier', ('No','Yes'))
       head_airbag = st.selectbox('Head Airbag', ('No','Yes'))
       side_airbags = st.selectbox('Side Airbags', ('No','Yes'))
       rear_airbags = st.selectbox('Rear Airbags', ('No','Yes'))
       drowsiness = st.selectbox('Driver Drowsiness Detection', ('No','Yes'))
       
       st.subheader('Door Count')
       door2 = st.selectbox('2', ('No','Yes'))
       door3 = st.selectbox('3', ('No','Yes'))
       door4 = st.selectbox('4', ('No','Yes'))
       door5 = st.selectbox('5', ('No','Yes'))
       door6 = st.selectbox('6', ('No','Yes')) 
       
       
    

    
    
    predict = st.button('Predict')

    if predict:  
           
        data = {
        
        'mileage': mileage,
        'warranty': 'No',
        'first_registration': 2022-age,
        'general_inspection': general_inspection,
        'non_smoker_vehicle': 'No',
        'Power': power,
        'engine_size': engine_size,
        'cylinders': cylinders,
        'empty_weight': empty_weight,
        'co2_emissions': carbon_emission(),
        'paint': 'No',
        'make_model': make_model1,
        'fuel_con_comb': 9,
        'bdt_Compact': compact,
        'bdt_Convertible': convertable,
        'bdt_Coupe': 'No',
        'bdt_Off-Road/Pick-up': off_road,
        'bdt_Panel van': 'No',
        'bdt_Sedan': sedan,
        'bdt_Station wagon': station,
        'bdt_Transporter': transporter,
        'bdt_Van': 'No',
        'gb_Automatic': automatic,
        'gb_Manual': manual,
        'gb_Semi-automatic': semi_auto,
        'age': age,
        'cc_360° camera': 'No',
        'cc_Air conditioning': air_cond,
        'cc_Air suspension': 'No',
        'cc_Armrest': 'Yes',
        'cc_Automatic climate control': 'No',
        'cc_Automatic climate control 2 zones': 'No',
        'cc_Automatic climate control 3 zones': 'No',
        'cc_Automatic climate control 4 zones': 'No',
        'cc_Auxiliary heating': 'No',
        'cc_Cruise control': cruis,
        'cc_Electric backseat adjustment': 'No',
        'cc_Electric tailgate': 'No',
        'cc_Electrical side mirrors': mirrors,
        'cc_Electrically adjustable seats': 'No',
        'cc_Electrically heated windshield': 'No',
        'cc_Fold flat passenger seat': 'No',
        'cc_Heads-up display': 'No',
        'cc_Heated steering wheel': 'No',
        'cc_Hill Holder': 'No',
        'cc_Keyless central door lock': door_lock,
        'cc_Leather seats': 'No',
        'cc_Leather steering wheel': 'No',
        'cc_Light sensor': 'No',
        'cc_Lumbar support': 'No',
        'cc_Massage seats': 'No',
        'cc_Multi-function steering wheel': 'No',
        'cc_Navigation system': 'No',
        'cc_Panorama roof': 'No',
        'cc_Park Distance Control': 'No',
        'cc_Parking assist system camera': 'No',
        'cc_Parking assist system self-steering': 'No',
        'cc_Parking assist system sensors front': 'No',
        'cc_Parking assist system sensors rear': 'No',
        'cc_Power windows': windows,
        'cc_Rain sensor': 'No',
        'cc_Seat heating': seat_heat,
        'cc_Seat ventilation': seat_ventilation,
        'cc_Sliding door left': 'No',
        'cc_Sliding door right': 'No',
        'cc_Split rear seats': 'No',
        'cc_Start-stop system': 'No',
        'cc_Sunroof': sunroof,
        'cc_Tinted windows': 'No',
        'cc_Wind deflector': wind_deflector,
        'type_Antique/Classic': 'No',
        'type_Demonstration': 'No',
        'type_New': new,
        'type_Pre-registered': 'No',
        'type_Used': used,
        'cyl_1': 'No',
        'cyl_10': 'No',
        'cyl_12': 'No',
        'cyl_2': 'No',
        'cyl_3': 'No',
        'cyl_4': 'No',
        'cyl_5': 'No',
        'cyl_6': 'No',
        'cyl_7': 'No',
        'cyl_8': 'No',
        'doors_1': 'No',
        'doors_2': door2,
        'doors_3': door3,
        'doors_4': door4,
        'doors_5': door5,
        'doors_6': door6,
        'clr_Beige': 'No',
        'clr_Black': 'No',
        'clr_Blue': 'No',
        'clr_Bronze': 'No',
        'clr_Brown': 'No',
        'clr_Gold': 'No',
        'clr_Green': 'No',
        'clr_Grey': 'No',
        'clr_Orange': 'No',
        'clr_Red': 'No',
        'clr_Silver': 'No',
        'clr_Violet': 'No',
        'clr_White': 'No',
        'clr_Yellow': 'No',
        'grs_1': 'No',
        'grs_2': 'No',
        'grs_3': 'No',
        'grs_4': 'No',
        'grs_5': 'No',
        'grs_6': 'No',
        'grs_7': 'No',
        'grs_8': 'No',
        'grs_9': 'No',
        'mileage_digitized': 6,
        'ext_All season tyres': 'No',
        'ext_Alloy wheels': 'No',
        'ext_Ambient lighting': 'No',
        'ext_Automatically dimming interior mirror': 'No',
        'ext_Awning': 'No',
        'ext_Biodiesel conversion': 'No',
        'ext_Cargo barrier': 'No',
        'ext_Catalytic Converter': 'No',
        'ext_E10-enabled': 'No',
        'ext_Electronic parking brake': 'No',
        'ext_Emergency tyre': 'No',
        'ext_Emergency tyre repair kit': 'No',
        'ext_Handicapped enabled': 'No',
        'ext_Headlight washer system': 'No',
        'ext_Range extender': 'No',
        'ext_Right hand drive': 'No',
        'ext_Roof rack': 'No',
        'ext_Shift paddles': 'No',
        'ext_Ski bag': 'No',
        'ext_Sliding door': 'No',
        'ext_Smokers package': 'No',
        'ext_Spare tyre': 'No',
        'ext_Spoiler': 'No',
        'ext_Sport package': 'No',
        'ext_Sport seats': 'No',
        'ext_Sport suspension': 'No',
        'ext_Steel wheels': 'No',
        'ext_Summer tyres': 'No',
        'ext_Touch screen': 'No',
        'ext_Trailer hitch': 'No',
        'ext_Tuned car': 'No',
        'ext_Voice Control': 'No',
        'ext_Winter package': 'No',
        'ext_Winter tyres': 'No',
        'em_Android Auto': 'No',
        'em_Apple CarPlay': 'No',
        'em_Bluetooth': 'No',
        'em_CD player': 'No',
        'em_Digital cockpit': 'No',
        'em_Digital radio': 'No',
        'em_Hands-free equipment': 'No',
        'em_Induction charging for smartphones': 'No',
        'em_Integrated music streaming': 'No',
        'em_MP3': 'No',
        'em_On-board computer': 'No',
        'em_Radio': 'No',
        'em_Sound system': 'No',
        'em_Television': 'No',
        'em_USB': 'No',
        'em_WLAN / WiFi hotspot': 'No',
        'dt_4WD': 'No',
        'dt_Front': 'No',
        'dt_Rear': 'No',
        'fuelt_CNG': 'No',
        'fuelt_Diesel': diesel,
        'fuelt_Electric': electric,
        'fuelt_Electric/Diesel': 'No',
        'fuelt_Electric/Gasoline': 'No',
        'fuelt_Ethanol': 'No',
        'fuelt_Gasoline': gasoline,
        'fuelt_Hydrogen': 'No',
        'fuelt_LPG': lpg,
        'up_Alcantara': 'No',
        'up_Cloth': 'No',
        'up_Full leather': 'No',
        'up_Part leather': 'No',
        'up_Velour': 'No',
        'upClr_Beige': 'No',
        'upClr_Black': 'No',
        'upClr_Blue': 'No',
        'upClr_Brown': 'No',
        'upClr_Green': 'No',
        'upClr_Grey': 'No',
        'upClr_Orange': 'No',
        'upClr_Red': 'No',
        'upClr_White': 'No',
        'upClr_Yellow': 'No',
        'seats_Beige': 'No',
        'seats_Black': 'No',
        'seats_Blue': 'No',
        'seats_Brown': 'No',
        'seats_Green': 'No',
        'seats_Grey': 'No',
        'seats_Orange': 'No',
        'seats_Red': 'No',
        'seats_White': 'No',
        'seats_Yellow': 'No',
        'EmClass_Euro 1': 'No',
        'EmClass_Euro 2': 'No',
        'EmClass_Euro 3': 'No',
        'EmClass_Euro 4': 'No',
        'EmClass_Euro 5': 'No',
        'EmClass_Euro 6': 'No',
        'ss_ABS': 'No',
        'ss_Adaptive Cruise Control': 'No',
        'ss_Adaptive headlights': 'No',
        'ss_Alarm system': 'No',
        'ss_Bi-Xenon headlights': 'No',
        'ss_Blind spot monitor': 'No',
        'ss_Central door lock': 'No',
        'ss_Central door lock with remote control': 'No',
        'ss_Daytime running lights': 'No',
        'ss_Distance warning system': 'No',
        'ss_Driver drowsiness detection': drowsiness,
        'ss_Driver-side airbag': 'No',
        'ss_Electronic stability control': 'No',
        'ss_Emergency brake assistant': 'No',
        'ss_Emergency system': 'No',
        'ss_Fog lights': fog_lights,
        'ss_Full-LED headlights': 'No',
        'ss_Glare-free high beam headlights': 'No',
        'ss_Head airbag': head_airbag,
        'ss_High beam assist': 'No',
        'ss_Immobilizer': immobilizier,
        'ss_Isofix': 'No',
        'ss_LED Daytime Running Lights': 'No',
        'ss_LED Headlights': 'No',
        'ss_Lane departure warning system': 'No',
        'ss_Laser headlights': 'No',
        'ss_Night view assist': 'No',
        'ss_Passenger-side airbag': 'No',
        'ss_Power steering': 'No',
        'ss_Rear airbag': rear_airbags,
        'ss_Side airbag': side_airbags,
        'ss_Speed limit control system': 'No',
        'ss_Tire pressure monitoring system': 'No',
        'ss_Traction control': 'No',
        'ss_Traffic sign recognition': 'No',
        'ss_Xenon headlights': 'No',

                    }
      
      
   
        features_df = pd.DataFrame.from_dict([data])
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write('Overview of input is shown below')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(features_df)
    
        preprocess_df = preprocess(features_df, 'Online')

        prediction = model.predict(preprocess_df)
    
        price = int(prediction)
        
        price = '💰💰''$$'+str(price)+'$$''💰💰'

        st.success(price)
        st.stop()
    
    else:
        st.subheader("Dataset upload")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file,encoding= 'utf-8')
            #Get overview of data
            st.write(data.head())
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            #Preprocess inputs
            preprocess_df = preprocess(data, "Batch")
            if st.button('Predict', key='1'):
               #Get batch prediction
                prediction = model.predict(preprocess_df)
                price = prediction
                prediction_df = pd.DataFrame(price, columns=["Predictions"])
                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.subheader('Prediction')
                st.write(prediction_df)
                

                
    
    
    
       

if __name__ == '__main__':
        main()
       