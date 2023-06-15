import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# import warnings
# warnings.filterwarnings("ignore")


st.title("PENAMBANGAN DATA C")

data_set_description, upload_data, preprocessing, modeling, implementation = st.tabs(["Data Set Description", "Data", "Preprocessing", "Modeling", "Implementation"])

with data_set_description:
    st.write("##### Nama  : Ahlaqul Karimah M ")
    st.write("##### Nim   : 200411100008 ")
    st.write("##### Kelas : Penambangan Data C ")
    st.write("""# Data Set Description """)
    st.write("###### Data Set Ini Adalah : Classification on Fish market (Klasifikasi dipasar ikan) ")
    st.write("###### Sumber Data Set dari Kaggle : https://www.kaggle.com/code/hongsean/tutorial-knn-classification-on-fish-market/notebook")
    st.write("""###### Penjelasan setiap kolom : """)
    st.write("""1. Species (Jenis ) :
    
    jenis :Spesies atau jenis adalah suatu peringkat taksonomi yang dipakai dalam klasifikasi biologis untuk merujuk pada satu atau beberapa kelompok individu makhluk hidup (populasi) yang serupa dan dapat saling membuahi satu sama lain di dalam kelompoknya (saling berbagi gen) sehingga menghasilkan keturunan yang fertil (subur).
    ikan : Ikan adalah anggota vertebrata poikilotermik (berdarah dingin)[1] yang hidup di air dan bernapas dengan insang. Ikan merupakan kelompok vertebrata yang paling beraneka ragam dengan jumlah spesies lebih dari 27,000 di seluruh dunia. Secara taksonomi, ikan tergolong kelompok paraphyletic yang hubungan kekerabatannya masih diperdebatkan; biasanya ikan dibagi menjadi ikan tanpa rahang (kelas Agnatha, 75 spesies termasuk lamprey dan ikan hag), ikan bertulang rawan (kelas Chondrichthyes, 800 spesies termasuk hiu dan pari), dan sisanya tergolong ikan bertulang keras (kelas Osteichthyes).
    """)
    st.write("""2. weight (Bobot) :

    bobot : bobot yang di ada di dataset
    """)
    st.write("""3. length1 (Panjang1) :

    panjang1 : panjang ikan yang ada  di dataset
    """)
    st.write("""4. length1 (Panjang2) :

    panjang2 : panjang ikan yang ada  di dataset
    """)
    st.write("""5. length2 (Panjang3) :

    panjang3 : panjang ikan yang ada  di dataset
    """)
    st.write("""6. height (Tinggi) :

    tinggi : tinggi ikan yang ada di sataset
    """)
    st.write("""7. width (Lrbar) :

    Output (keluaran)
    """)
    st.write("""Menggunakan Kolom (input) :

    precipitation
    tempmax * tempmin
    wind
    """)
    st.write("""Mengklasifikasi ikan di pasar   (output) :
    
    1. Bream    : istilah umum bagi sejumlah spesies ikan air tawar dan ikan laut dari beragam genus yang meliputi: Abramis (misalnya A. brama, terkadang disebut bream air tawar).
    2. Parkki   : -
    3. Perch    : merupakan spesies ikan yang berwarna perak dengan semburat biru. Mereka memiliki mata hitam gelap yang khas dengan cincin luar berwarna kuning cerah.
    4. Pike     : ikan dengan bentuk Mulut dan hidung berbentuk moncong dengan gigi terlihat mencolok banyak di rahang. Sirip kekuningan atau coklat kemerahan, punggung, dubur, dan ekor dengan bintik-bintik gelap menyebar. Biasanya ditemukan di air tawar meskipun baru-baru ditemukan hidup dalam air dengan kadar garam yang sedikit lebih rendah dari laut.
    5. Roach    : merupakan ikan air tawar yang berasal dari perairan di wilayah Eropa dan Asia. Ikan jenis ini mampu hidup di perairan payau. Rutilus ritilus adalah jenis ikan omnivora yang tersebar luas di wilayah Eropa dan menjadi invasif di wilayah Irlandia dan Italia.
    6. Smelt    : sejenis ikan kecil yang dipakai sebagai umpan.
    7. Whitefish: adalah satu istilah perikanan yang merujuk kepada beberapa spesies ikan demersal dengan sirip, khususnya kod (Gadus morhua), whiting (Merluccius bilinearis), dan haddock (Melanogrammus aeglefinus).
    
    """)
    st.write("###### Aplikasi ini untuk : Classification on Fish market (Klasifikasi ikan di Pasar) ")
    st.write("###### Source Code Aplikasi ada di Github anda bisa acces di link : https://github.com/08-Ahlaqul-Karimah/project-data-mining ")

with upload_data:
    # uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    # for uploaded_file in uploaded_files:
    #     df = pd.read_csv(uploaded_file)
    #     st.write("Nama File Anda = ", uploaded_file.name)
    #     st.dataframe(df)
    df = pd.read_csv('https://raw.githubusercontent.com/08-Ahlaqul-Karimah/machine-Learning/main/Fish.csv')
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
    df = df.drop(columns=['Length2','Length3'])
    #Mendefinisikan Varible X dan Y
    X = df[['Weight','Length1','Height','Width']]
    y = df['Species'].values
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
    dumies = pd.get_dummies(df.Species).columns.values.tolist()
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

    # st.subheader("""Normalisasi Data""")
    # st.write("""Rumus Normalisasi Data :""")
    # st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
    # st.markdown("""
    # Dimana :
    # - X = data yang akan dinormalisasi atau data asli
    # - min = nilai minimum semua data asli
    # - max = nilai maksimum semua data asli
    # """)
    # df.weather.value_counts()
    # df = df.drop(columns=["date"])
    # #Mendefinisikan Varible X dan Y
    # X = df.drop(columns=['weather'])
    # y = df['weather'].values
    # df_min = X.min()
    # df_max = X.max()

    # #NORMALISASI NILAI X
    # scaler = MinMaxScaler()
    # #scaler.fit(features)
    # #scaler.transform(features)
    # scaled = scaler.fit_transform(X)
    # features_names = X.columns.copy()
    # #features_names.remove('label')
    # scaled_features = pd.DataFrame(scaled, columns=features_names)

    # #Save model normalisasi
    # from sklearn.utils.validation import joblib
    # norm = "normalisasi.save"
    # joblib.dump(scaled_features, norm) 


    # st.subheader('Hasil Normalisasi Data')
    # st.write(scaled_features)

with modeling:
    training, test = train_test_split(scaled_features,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
    training_label, test_label = train_test_split(y, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing
    with st.form("modeling"):
        st.subheader('Modeling')
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        naive = st.checkbox('Gaussian Naive Bayes')
        k_nn = st.checkbox('K-Nearest Neighboor')
        destree = st.checkbox('Decission Tree')
        submitted = st.form_submit_button("Submit")

        # NB
        GaussianNB(priors=None)

        # Fitting Naive Bayes Classification to the Training set with linear kernel
        gaussian = GaussianNB()
        gaussian = gaussian.fit(training, training_label)

        # Predicting the Test set results
        y_pred = gaussian.predict(test)
    
        y_compare = np.vstack((test_label,y_pred)).T
        gaussian.predict_proba(test)
        gaussian_akurasi = round(100 * accuracy_score(test_label, y_pred))
        # akurasi = 10

        #Gaussian Naive Bayes
        # gaussian = GaussianNB()
        # gaussian = gaussian.fit(training, training_label)

        # probas = gaussian.predict_proba(test)
        # probas = probas[:,1]
        # probas = probas.round()

        # gaussian_akurasi = round(100 * accuracy_score(test_label,probas))

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
            if naive :
                st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(gaussian_akurasi))
            if k_nn :
                st.write("Model KNN accuracy score : {0:0.2f}" . format(knn_akurasi))
            if destree :
                st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(dt_akurasi))
        
        grafik = st.form_submit_button("Grafik akurasi semua model")
        if grafik:
            data = pd.DataFrame({
                'Akurasi' : [gaussian_akurasi, knn_akurasi, dt_akurasi],
                'Model' : ['Gaussian Naive Bayes', 'K-NN', 'Decission Tree'],
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
        Weight = st.number_input('Masukkan berat ikan (Weight) : ')
        Length1 = st.number_input('Masukkan panjang vertikal ikan (Length1) : ')
        Height = st.number_input('Masukkan tinggi ikan (Height) : ')
        Width = st.number_input('Masukkan lebar ikan (Width) : ')
        
        model = st.selectbox('Pilihlah model yang akan anda gunakan untuk melakukan prediksi?',
                ('Gaussian Naive Bayes', 'K-NN', 'Decision Tree'))

        prediksi = st.form_submit_button("Submit")
        if prediksi:
            inputs = np.array([
                Weight,
                Length1,
                Height,
                Width
            ])

            df_min = X.min()
            df_max = X.max()
            input_norm = ((inputs - df_min) / (df_max - df_min))
            input_norm = np.array(input_norm).reshape(1, -1)

            if model == 'Gaussian Naive Bayes':
                mod = gaussian
            if model == 'K-NN':
                mod = knn 
            if model == 'Decision Tree':
                mod = dt

               
            input_pred = mod.predict(input_norm)


            st.subheader('Hasil Prediksi')
            st.write('Menggunakan Pemodelan :', model)

            st.write(input_pred)
