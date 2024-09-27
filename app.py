import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import streamlit as st
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from streamlit_echarts import st_echarts

 
st.set_page_config(page_title="Rekomendasi Skripsi", page_icon=":robot_face:", layout="wide", initial_sidebar_state="expanded")

def p_title(title):
    st.markdown(f'<h3 style="text-align: left; color:#F63366; font-size:28px;">{title}</h3>', unsafe_allow_html=True)

st.sidebar.header('Menu SIREKTSI !')
st.sidebar.image("logo.png", use_column_width=True)
nav = st.sidebar.radio('',['🏠 Homepage', '📄 Data Skripsi', '📊 Grafik Dosen Pembimbing', '🔍 Rekomendasi Skripsi'])
st.sidebar.write('___')

if nav == '🏠 Homepage':
    # st.title('Sistem Rekomendasi Topik Skripsi Prodi Sistem Informasi')
    st.markdown("<h1 style='text-align: center; color: white; font-size:28px;'>Welcome to SIREKTSI! 👋</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: white; font-size:28px;'>Sistem Rekomendasi Topik Skripsi Prodi Sistem Informasi</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; font-size:56px;'<p>📚🔍</p></h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: grey; font-size:20px;'>Memuat informasi Seluruh Data Skripsi, Grafik Dosen Pembimbing Skripsi, Memberikan Rekomendasi Topik Skripsi!</h3>", unsafe_allow_html=True)
    st.markdown('___')
    st.write(':point_left: Gunakan menu di samping kiri ini (click on > if closed).')
    st.markdown('___')
    st.markdown("<h3 style='text-align: left; color:#F63366; font-size:18px;'><b>What is this App about?<b></h3>", unsafe_allow_html=True)
    st.write("Sebuah sistem rekomendasi dengan menggunakan metode content based filtering dan cosine similarity .")
    st.write("SIREKTSI :robot_face:, adalah sebuah sistem rekomendasi dengan memanfaatkan abstract skripsi yang serupa dengan yang diminati oleh user, juga dapat melihat data skripsi seluruh mahasiswa prodi Sistem Infromasi dari para alumni, dan juga dapat melihat grafik dosen pembimbing skripsi prodi Sistem Informasi!")     
    st.markdown("<h3 style='text-align: left; color:#F63366; font-size:18px;'><b>Who is this App for?<b></h3>", unsafe_allow_html=True)
    st.write("Sistem rekomendasi ini di tujukan kepada seluruh mahasiswa prodi Sistem Informasi pada Universitas Trunojoyo Madura yang ingin mencari ide atau topik penelitian skripsinya :smiley:")

if nav == '📄 Data Skripsi':

    data_skripsi = pd.read_csv('cleaned_data_skripsi.csv')

    # Filter hanya baris yang memiliki abstrak
    data_skripsi = data_skripsi[data_skripsi['Abstrak'].notnull()]
    data_skripsi.reset_index(drop=True, inplace=True)

    # Tampilkan editor untuk data skripsi
    st.title("Data Skripsi Prodi Sistem Informasi")

    edited_data = st.data_editor(
        data_skripsi[['No.', 'Link Skripsi', 'Judul Skripsi', 'Penulis', 'NIM', 'Tahun', 'Abstrak', 'Dospem1', 'Dospem2']],
        num_rows="dynamic",  # Menambah baris secara dinamis
        key="editor"
    )

    if st.button("Simpan Perubahan"):
        # Menggabungkan data yang diedit dengan kolom lain yang tidak diubah
        for col in data_skripsi.columns:
            if col not in edited_data.columns:
                edited_data[col] = data_skripsi[col]
        
        # Menyimpan kembali data yang telah diperbarui ke file CSV
        edited_data.to_csv('cleaned_data_skripsi.csv', index=False)
        st.success("Perubahan telah disimpan ke cleaned_data_skripsi.csv!")

if nav == '📊 Grafik Dosen Pembimbing':
    data_skripsi = pd.read_csv('cleaned_data_skripsi.csv')

    #FILTER PER DOSEN PEMBIMBING
    df = pd.DataFrame(data_skripsi[['Judul Skripsi', 'Dospem1', 'Dospem2', 'Abstrak']])
    all_tags = set(df['Dospem1']).union(set(df['Dospem2']))
    st.title('Filter Skripsi Berdasarkan Dosen Pembimbing')
    selected_tags = st.multiselect('Pilih Dosen Pembimbing:', options=list(all_tags))
    if selected_tags:
        filtered_df = df[df.apply(lambda row: any(dospem in selected_tags for dospem in [row['Dospem1'], row['Dospem2']]), axis=1)]
    else:
        filtered_df = df  # Jika tidak ada tag yang dipilih, tampilkan seluruh data
    st.write('Hasil Skripsi:')
    st.dataframe(filtered_df)

    # Hitung jumlah kemunculan setiap dosen sebagai Dosen Pembimbing 1
    dospem1_count = data_skripsi['Dospem1'].value_counts()
    # Hitung jumlah kemunculan setiap dosen sebagai Dosen Pembimbing 2
    dospem2_count = data_skripsi['Dospem2'].value_counts()
    total_dospem_count = dospem1_count.add(dospem2_count, fill_value=0)


    #PIECHART GABUNGAN JUMLAH TOTAL SKRIPSI YANG TELAH DI BIMBING DOSEN
    total_dospem_counts = dospem1_count.add(dospem2_count, fill_value=0)
    # Format data untuk pie chart
    pie_data = [{"value": int(count), "name": dospem} for dospem, count in total_dospem_counts.items()]
    # Pie chart options
    options = {
        "backgroundColor": "white",
        "title": {
            "text": "Total Skripsi",
            "subtext": "Dosen Pembimbing",
            "left": "center",
            "top": "top",
            "padding": [10, 10, 10, 10]
        },
        "tooltip": {"trigger": "item"},
        "legend": {
            "orient": "horizontal",  # Legend dengan orientasi horizontal
            "top": "50",  # Legend di bawah title (gunakan 'px' untuk penempatan yang tepat)
            "left": "center",  # Posisikan legend di tengah
            "padding": [10, 5, 10, 5],
        },
        "series": [
            {
                "name": "Jumlah Skripsi",
                "type": "pie",
                "radius": ["50%", "100%"],  # Membuat pie chart menjadi donut
                "avoidLabelOverlap": False,
                "label": {
                    "show": True,
                    "position": "outside",  # Label berada di luar pie chart
                    "formatter": "{b}: {c} ({d}%)",  # Format label menampilkan nama, jumlah, dan persentase
                },
                "labelLine": {
                    "show": True,  # Garis penghubung ke label di luar pie chart
                },
                "emphasis": {
                    "itemStyle": {
                        "shadowBlur": 10,
                        "shadowOffsetX": 0,
                        "shadowColor": "rgba(0, 0, 0, 0.5)",
                    }
                },
                "data": pie_data,
            }
        ],
        "responsive": True,  # Menambahkan logika responsif
        "media": [
            {
                "query": {
                    "maxWidth": 900,
                },
                "option": {
                    "legend": {
                        "orient": "horizontal",  # Legend menjadi vertikal di layar kecil
                        "type": "scroll",  # Tambahkan scroll untuk legend
                        "left": "left",  # Legend tetap di tengah
                        "top": "bottom",  # Legend di bagian bawah chart
                        "padding": [10, 10, 10, 10],
                    },
                    "series": [
                        {
                            "top": "10%",  # Jarak antara chart dan legend lebih besar
                            "radius": ["50%", "100%"],  # Ukuran pie chart lebih kecil
                        }
                    ]
                }

            },
            {
                "query": {
                    "minWidth": 901,  # Saat lebar layar lebih besar dari 900px
                },
                "option": {
                    "legend": {
                        "orient": "horizontal",  # Legend tetap horizontal di layar besar
                        "type": "scroll",  # Mengaktifkan scroll untuk legend
                        "top": "50",  # Legend di bawah title
                    },
                    "series": [
                        {
                            "top": "20%",  # Ukuran pie chart lebih besar pada layar besar
                            "radius": ["50%", "100%"]
                        }
                    ]
                }
            }
        ]
    }

    # Tampilkan pie chart menggunakan st_echarts
    st.markdown("### Total Skripsi Dosen Pembimbing")
    st.markdown("Pilih sebuah dosen pembimbing di legenda untuk melihat detailnya.")
    events = {
        "legendselectchanged": "function(params) { return params.selected }",
    }
    s = st_echarts(options=options, events=events, height="600px", key="render_pie_events")

    # Plot grafik batang untuk Dosen Pembimbing 1
    st.write("### Distribusi Dosen Pembimbing 1")
    fig, ax = plt.subplots(figsize=(10, 6))
    dospem1_count.plot(kind='bar', color='lightcoral', ax=ax)
    # Judul dan label sumbu untuk grafik Dosen Pembimbing 1
    ax.set_title('Distribusi Dosen Pembimbing 1', fontsize=14)
    ax.set_xlabel('Nama Dosen Pembimbing 1', fontsize=12)
    ax.set_ylabel('Jumlah Skripsi Dibimbing', fontsize=12)
    # Menambahkan keterangan jumlah skripsi di atas setiap batang untuk Dospem 1
    for index, value in enumerate(dospem1_count):
        ax.text(index, value + 0.1, str(value), ha='center', fontsize=10)

    # Menampilkan grafik di Streamlit
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    # Plot grafik batang untuk Dosen Pembimbing 2
    st.write("### Distribusi Dosen Pembimbing 2")
    fig, ax = plt.subplots(figsize=(10, 6))
    dospem2_count.plot(kind='bar', color='lightblue', ax=ax)
    # Judul dan label sumbu untuk grafik Dosen Pembimbing 2
    ax.set_title('Distribusi Dosen Pembimbing 2', fontsize=14)
    ax.set_xlabel('Nama Dosen Pembimbing 2', fontsize=12)
    ax.set_ylabel('Jumlah Skripsi Dibimbing', fontsize=12)
    # Menambahkan keterangan jumlah skripsi di atas setiap batang untuk Dospem 2
    for index, value in enumerate(dospem2_count):
        ax.text(index, value + 0.1, str(value), ha='center', fontsize=10)

    # Menampilkan grafik di Streamlit
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

if nav == '🔍 Rekomendasi Skripsi':

    # Fungsi untuk membersihkan teks
    def text_clean(text):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        sastrawi = StopWordRemoverFactory()
        stopworda = sastrawi.get_stop_words()
        clean_spcl = re.compile('[/(){}\[\]\|@,;]')
        clean_symbol = re.compile('[^0-9a-z #+_]')
        text = text.lower()
        text = clean_spcl.sub(' ', text)
        text = clean_symbol.sub('', text)
        text = stemmer.stem(text)
        text = ' '.join(word for word in text.split() if word not in stopworda)
        return text

    def recommendations(keyword, top=10):
        rekomendasi = []
        cleaned_keyword = text_clean(keyword)

        # TF-IDF untuk input keyword
        tfidf_keyword_judul = tfidf_vectorizer_judul.transform([cleaned_keyword])
        tfidf_keyword_abstrak = tfidf_vectorizer_abstrak.transform([cleaned_keyword])
        
        # Menghitung cosine similarity
        scores_judul = cosine_similarity(tfidf_matrix_judul, tfidf_keyword_judul).flatten()
        scores_abstrak = cosine_similarity(tfidf_matrix_abstrak, tfidf_keyword_abstrak).flatten()
        
        # Menjumlahkan skor dari judul dan abstrak
        combined_scores = scores_judul + scores_abstrak
        
        # Mengurutkan berdasarkan skor tertinggi
        sorted_indexes = np.argsort(combined_scores)[::-1]
        
        # Mendapatkan rekomendasi berdasarkan top skor
        for i in sorted_indexes[:top]:
            judul = data_skripsi.iloc[i]['Judul Skripsi']
            link = data_skripsi.iloc[i]['Link Skripsi']
            penulis = data_skripsi.iloc[i]['Penulis']
            nim = data_skripsi.iloc[i]['NIM']
            abstrak = data_skripsi.iloc[i]['Abstrak']
            score = combined_scores[i]
            dospem1 = data_skripsi.iloc[i]["Dospem1"]
            dospem2 = data_skripsi.iloc[i]["Dospem2"]
            rekomendasi.append((judul, link, penulis, nim, abstrak, score, dospem1, dospem2))
        
        return rekomendasi

    # Load data skripsi yang sudah dibersihkan
    data_skripsi = pd.read_csv('cleaned_data_skripsi.csv')

    # Membuat TF-IDF untuk judul dan abstrak
    tfidf_vectorizer_judul = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0.00, max_df=0.85, sublinear_tf=True)
    tfidf_vectorizer_abstrak = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0.00, max_df=0.85, sublinear_tf=True)

    # Membuat matriks TF-IDF
    tfidf_matrix_judul = tfidf_vectorizer_judul.fit_transform(data_skripsi['cleaned_judul'])
    tfidf_matrix_abstrak = tfidf_vectorizer_abstrak.fit_transform(data_skripsi['cleaned_abstrak'])

    # Streamlit UI
    st.title('Sistem Rekomendasi Skripsi')


    with st.form(key='rekomendasi_form'):
        keyword = st.text_input("Masukkan keyword untuk mencari skripsi:")
        jumlah_rekomendasi = st.slider("Jumlah rekomendasi yang diinginkan:", 1, 20, 10)
        
        # Tombol untuk submit form
        submit_button = st.form_submit_button(label='Cari Rekomendasi')

    # Placeholder untuk menampilkan hasil rekomendasi
    results_placeholder = st.empty()

    # Jika form disubmit, tampilkan rekomendasi
    if submit_button:
        if keyword:
            results_placeholder.empty()

            hasil_rekomendasi = recommendations(keyword, jumlah_rekomendasi)

            with results_placeholder.container():
                st.write(f"Hasil rekomendasi skripsi yang mungkin Anda sukai berdasarkan '{keyword}':")
                
                # Menampilkan hasil dalam bentuk tabel
                for index, (judul, link, penulis, nim, abstrak, score, dospem1, dospem2) in enumerate(hasil_rekomendasi, 1):
                    st.write(f"### Rekomendasi ke-{index}:")
                    st.write(f"**Judul**: {judul}")
                    st.write(f"**Link**: {link}")
                    st.write(f"**Penulis**: {penulis} ({nim})")
                    st.write(f"**Abstrak**: {abstrak}")
                    st.write(f"**Score**: {score:.4f}")
                    st.write(f"**Dosen Pembimbing 1**: {dospem1}")
                    st.write(f"**Dosen Pembimbing 2**: {dospem2}")
                    st.markdown('___')
        else:
            st.warning("Masukkan keyword untuk mendapatkan rekomendasi.")

