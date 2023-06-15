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

st.title("Data PT Petrosea Tbk")

st.write("Name : Elmatia Dwi Uturiyah")
st.write("Nim : 200411100113")

data_set_description, data, preprocessing, modeling, implementation = st.columns(5)

with data_set_description:
    st.header("Data Set Description")
    st.write("PT Petrosea Tbk menyediakan teknik, konstruksi, pertambangan, dan layanan lainnya untuk sektor minyak dan gas, infrastruktur, industri dan manufaktur, dan utilitas di Indonesia dan internasional. Ini beroperasi melalui tiga segmen: Pertambangan, Layanan, dan Rekayasa dan Konstruksi. Segmen Pertambangan menyediakan jasa kontrak penambangan, termasuk pengupasan lapisan tanah penutup, pengeboran, peledakan, pengangkatan, pengangkutan, tambang, dan jasa rekanan tambang. Segmen Layanan menawarkan fasilitas basis pasokan dan layanan pelabuhan. Segmen Teknik dan Konstruksi menyediakan serangkaian layanan teknik, pengadaan, dan konstruksi, termasuk uji tuntas teknis, studi kelayakan, desain teknik, manajemen proyek, pengadaan dan logistik, penyewaan pabrik dan peralatan, serta layanan operasi dan komisioning. Segmen ini juga memasok tenaga perdagangan terampil. Perusahaan didirikan pada tahun 1972 dan berkantor pusat di Tangerang Selatan, Indonesia. PT Petrosea Tbk merupakan anak perusahaan dari PT Indika Energy Tbk. Per 28 Juli 2022, PT Petrosea Tbk beroperasi sebagai anak perusahaan PT Caraka Reksa Optima.")
    st.write("Sumber Data Set dari Finance yahoo.com : [https://github.com/asimcode2050/Asim-Code-Youtube-Channel-Code/blob/main/python/yahoo_finance.py](https://github.com/asimcode2050/Asim-Code-Youtube-Channel-Code/blob/main/python/yahoo_finance.py)")
    st.write("Source Code Aplikasi ada di Github anda bisa acces di link : [https://elmatiaaa.github.io/prosaindata/Uas_Kelompok.html](https://elmatiaaa.github.io/prosaindata/Uas_Kelompok.html)")

with data:
    st.header("Data")
    df = pd.read_csv('https://raw.githubusercontent.com/elmatiaaa/prosaindata/main/new.csv')
    st.dataframe(df)

with preprocessing:
    st.header("Preprocessing")
    st.subheader("Normalisasi Data")
    st.write("Rumus Normalisasi Data:")
    st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
    st.markdown("""
    Dimana:
    - X = data yang akan dinormalisasi atau data asli
    - min = nilai minimum semua data asli
    - max = nilai maksimum semua data asli
    """)

    df = df.drop(columns=['Date'])
    X = df[['Open', 'High', 'Low', 'Close', 'AdjClose']]
    y = df["Volume"].values
    df_min = X.min()
    df_max = X.max()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.subheader("Hasil Normalisasi Data")
    st.dataframe(scaled_features)

    st.subheader("Target Label")
    labels = pd.get_dummies(df.Volume).columns.values.tolist()
    labels = np.array(labels)
    labels_df = pd.DataFrame({f"{i+1}": [labels[i]] for i in range(len(labels))})
    st.dataframe(labels_df)

with modeling:
    st.header("Modeling")
    training, test, training_label, test_label = train_test_split(
        scaled_features, y, test_size=0.2, random_state=1
    )

    st.subheader("Model Selection")
    k_nn = st.checkbox("K-Nearest Neighbor")
    decision_tree = st.checkbox("Decision Tree")

    if k_nn or decision_tree:
        submitted = st.button("Submit")

        if submitted:
            if k_nn:
                K = 10
                knn = KNeighborsClassifier(n_neighbors=K)
                knn.fit(training, training_label)
                knn_predict = knn.predict(test)
                knn_accuracy = round(100 * accuracy_score(test_label, knn_predict))
                st.write("Model K-Nearest Neighbor accuracy score: {:.2f}".format(knn_accuracy))

            if decision_tree:
                dt = DecisionTreeClassifier()
                dt.fit(training, training_label)
                dt_pred = dt.predict(test)
                dt_accuracy = round(100 * accuracy_score(test_label, dt_pred))
                st.write("Model Decision Tree accuracy score: {:.2f}".format(dt_accuracy))

        if st.button("Grafik akurasi semua model"):
            models = []
            accuracies = []

            if k_nn:
                models.append("K-Nearest Neighbor")
                accuracies.append(knn_accuracy)

            if decision_tree:
                models.append("Decision Tree")
                accuracies.append(dt_accuracy)

            data = pd.DataFrame({"Model": models, "Akurasi": accuracies})

            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    x="Akurasi",
                    y=alt.Y("Model", sort="-x"),
                    color="Akurasi",
                    tooltip=["Akurasi", "Model"],
                )
                .properties(width=600, height=300)
                .interactive()
            )

            st.altair_chart(chart)

with implementation:
    st.header("Implementation")

    with st.form("my_form"):
        st.subheader("Implementasi")

        Open = st.number_input("Input Open:")
        High = st.number_input("Input High:")
        Low = st.number_input("Input Low:")
        Close = st.number_input("Input Close:")
        AdjClose = st.number_input("Input AdjClose:")
        model = st.selectbox(
            "Pilihlah model yang akan anda gunakan untuk melakukan prediksi?",
            ("K-Nearest Neighbor", "Decision Tree"),
        )
        submitted = st.form_submit_button("Submit")

        if submitted:
            inputs = np.array([Open, High, Low, Close, AdjClose])
            input_norm = (inputs - df_min) / (df_max - df_min)
            input_norm = input_norm.reshape(1, -1)

            if model == "K-Nearest Neighbor":
                prediction = knn.predict(input_norm)
            elif model == "Decision Tree":
                prediction = dt.predict(input_norm)

            st.subheader("Hasil Prediksi")
            st.write("Menggunakan Pemodelan:", model)
            st.write(prediction)
