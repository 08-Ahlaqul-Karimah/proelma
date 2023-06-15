import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# import warnings
# warnings.filterwarnings("ignore")


st.title("Web Apps - Star Dataset to Predict Stars Types")

st.write("================================================================================")

st.write("Name :Isnaini")
st.write("Nim  :200411100038")
st.write("Grade: Penambangan Data c")

data_set_description, data, preprocessing, modeling, implementation = st.tabs(["Data Set Description", "Data", "Preprocessing", "Modeling", "Implementation"])

with data_set_description:
    st.write("""# Data Set Description """)
    st.write("###### Data Set Ini Adalah : Predict Star Types ")
    st.write("###### Sumber Data Set dari Kaggle : https://www.kaggle.com/datasets/deepu1109/star-dataset")
    st.write("""Dalam dataset ini terdapat 59 data dan 7 kolom yaitu Temperature_(K),Luminosity(L/Lo),Radius(R/Ro),Absolute_magnitude(Mv),Star_type,Star_Color,Spectral_Class. 
    """)
    st.write("""###### DATASET INFO : """)
    st.write("""1. Ini adalah kumpulan data yang terdiri dari beberapa fitur bintang, diantaranya :  :
    Absolute Temperature (in K)
    Relative Luminosity (L/Lo)
    Relative Radius (R/Ro)
    Absolute Magnitude (Mv)
    Star Color (white,Red,Blue,Yellow,yellow-orange etc)
    Spectral Class (O,B,A,F,G,K,,M)
    Star Type *(Red Dwarf, Brown Dwarf, White Dwarf, Main Sequence , SuperGiants, HyperGiants)*
    Lo = 3.828 x 10^26 Watts (Avg Luminosity of Sun)
    Ro = 6.9551 x 10^8 m (Avg Radius of Sun)
    """)
    st.write("""2.Spectral Class :
    ini akan menjadi outputnya yaitu kelas scpectral.Dalam Aplikasi ini  akan emnghasilkan 7 prediksi  yaitu O,B,A,F,G,K,,M.
    """)
    st.write("""Memprediksi Spectral Class (output) :

    1. O
    2. B
    3. A 
    4. F
    5.G
    6.K
    7.M
    
    """)
    st.write("###### Aplikasi ini untuk : Star Dataset To Predict Star Types (Kumpulan Data Bintang Untuk Memprediksi Jenis Bintang) ")
    st.write("###### Source Code Aplikasi ada di Github anda bisa acces di link : https://github.com/135-ShintaNuriyatulMahmudiyah/PenambanganDataWeb ")
    st.write("###### Untuk Wa saya anda bisa hubungi nomer ini : http://wa.me/6285704097096 ")

with data:
    df = pd.read_csv('https://raw.githubusercontent.com/elmatiaaa/prosaindata/main/new.csv')
    st.dataframe(df)

with preprocessing:
    st.subheader("""Normalisasi Data""")
    st.write("""Rumus Normalisasi Data :""")
    st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
    st.markdown("""
    Dimana :
    - X = data yang akan dinormalisasi atau data asli
    - min = nilai minimum semua data asli
    - max = nilai maksimum semua data asli
    """)
    

    
    df = df.drop(columns=['Date'])
    #Mendefinisikan Varible X dan Y
    X = df[['Open','High','Low','Close','AdjClose']]
    y = df["Volume"].values
    df
    X
    df_min = X.min()
    df_max = X.max()
      #NORMALISASI NILAI X
    scaler = MinMaxScaler()
    #scaler.fit(features)
    #scaler.transform(features)
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    #features_names.remove('label')
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_features)

    st.subheader('Target Label')
    dumies = pd.get_dummies(df.Volume).columns.values.tolist()
    dumies = np.array(dumies)

    labels = pd.DataFrame({
        '1' : [dumies[0]],
        '2' : [dumies[1]],
        '3' : [dumies[2]],
        '4' : [dumies[3]],
        '5' : [dumies[4]],
        '6' : [dumies[5]],
        '7' : [dumies[6]],
        
    })

    st.write(labels)

   
with modeling:
    training, test = train_test_split(scaled_features,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
    training_label, test_label = train_test_split(y, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing

    with st.form("modeling"):
        st.subheader('Modeling')
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        k_nn = st.checkbox('K-Nearest Neighboor')
        destree = st.checkbox('Decission Tree')
        submitted = st.form_submit_button("Submit")

       
        #KNN
        K=10
        knn=KNeighborsClassifier(n_neighbors=K)
        knn.fit(training,training_label)
        knn_predict=knn.predict(test)

        knn_akurasi = round(100 * accuracy_score(test_label,knn_predict))

        #Decission Tree
        dt = DecisionTreeClassifier()
        dt.fit(training, training_label)
        # prediction
        dt_pred = dt.predict(test)
        #Accuracy
        dt_akurasi = round(100 * accuracy_score(test_label,dt_pred))

        if submitted :
            if k_nn :
                st.write("Model KNN accuracy score : {0:0.2f}" . format(knn_akurasi))
            if destree :
                st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(dt_akurasi))
        
        grafik = st.form_submit_button("Grafik akurasi semua model")
        if grafik:
            data = pd.DataFrame({
                'Akurasi' : [knn_akurasi, dt_akurasi],
                'Model' : [ 'K-NN', 'Decission Tree'],
            })

            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.X("Akurasi"),
                    alt.Y("Model"),
                    alt.Color("Akurasi"),
                    alt.Tooltip(["Akurasi", "Model"]),
                )
                .interactive()
            )
            st.altair_chart(chart,use_container_width=True)
            
  
with implementation:
    with st.form("my_form"):
        st.subheader("Implementasi") 
        Open = st.number_input('input Open : ')
        High = st.number_input('Input High : ')
        Low= st.number_input('Input Low : ')
        Close = st.number_input('Input Close : ')
        AdjClose = st.number_input('Input AdjClose : ')
        model = st.selectbox('Pilihlah model yang akan anda gunakan untuk melakukan prediksi?',
                ( 'K-NN', 'Decision Tree'))

        prediksi = st.form_submit_button("Submit")
        if prediksi:
            inputs = np.array([
                'Open',
                'High',
                'Low',
                'Close',
                'AdjClose'
               
            ])

            df_min = X.min()
            df_max = X.max()
            input_norm = ((inputs - df_min) / (df_max - df_min))
            input_norm = np.array(input_norm).reshape(1, -1)
            if model == 'K-NN':
                mod = knn 
            if model == 'Decision Tree':
                mod = dt

            input_pred = mod.predict(input_norm)


            st.subheader('Hasil Prediksi')
            st.write('Menggunakan Pemodelan :', model)

            st.write(input_pred)
