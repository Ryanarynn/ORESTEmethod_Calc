import streamlit as st
import numpy as np
import pandas as pd
import io
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Custom CSS untuk desain futuristik
st.markdown("""
    <style>
    /* Palet warna: Neon biru (#00ddeb), ungu (#c084fc), hitam (#0a0a0a) */
    .main {
        background: linear-gradient(145deg, #0a0a0a, #1e1e2f);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 0 20px rgba(0, 221, 235, 0.3);
    }
    .stApp {
        background: transparent;
    }
    .card {
        background: rgba(30, 30, 47, 0.9);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 0 15px rgba(192, 132, 252, 0.2);
        margin-bottom: 25px;
        transition: all 0.3s ease;
        border: 1px solid rgba(0, 221, 235, 0.3);
    }
    .card:hover {
        box-shadow: 0 0 20px rgba(192, 132, 252, 0.4);
        transform: translateY(-3px);
    }
    .stButton>button {
        background: linear-gradient(90deg, #00ddeb, #c084fc);
        color: white;
        border-radius: 10px;
        padding: 15px 30px;
        font-weight: bold;
        border: none;
        transition: all 0.4s ease;
        box-shadow: 0 0 10px rgba(0, 221, 235, 0.5);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #00c4d4, #a855f7);
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(192, 132, 252, 0.7);
    }
    .stTextInput>div>input, .stNumberInput>div>input {
        background: rgba(255, 255, 255, 0.05);
        color: #e0e0e0;
        border-radius: 8px;
        border: 1px solid rgba(0, 221, 235, 0.3);
        padding: 12px;
        transition: all 0.3s ease;
    }
    .stTextInput>div>input:focus, .stNumberInput>div>input:focus {
        border-color: #c084fc;
        box-shadow: 0 0 8px rgba(192, 132, 252, 0.5);
    }
    h1 {
        color: #00ddeb;
        font-family: 'Orbitron', sans-serif;
        text-align: center;
        text-shadow: 0 0 10px rgba(0, 221, 235, 0.7);
    }
    h2 {
        color: #c084fc;
        font-family: 'Orbitron', sans-serif;
        border-bottom: 2px solid #00ddeb;
        padding-bottom: 8px;
        text-shadow: 0 0 5px rgba(192, 132, 252, 0.5);
    }
    .stDataFrame {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(0, 221, 235, 0.3);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 0 10px rgba(0, 221, 235, 0.2);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(145deg, #0a0a0a, #1e1e2f);
        color: #e0e0e0;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 0 15px rgba(0, 221, 235, 0.3);
    }
    .sidebar h3 {
        color: #00ddeb;
        font-family: 'Orbitron', sans-serif;
        text-shadow: 0 0 5px rgba(0, 221, 235, 0.5);
    }
    .sidebar .stMarkdown {
        color: #b0b0b0;
    }
    .stAlert {
        background: rgba(255, 75, 75, 0.1);
        color: #ff4b4b;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        box-shadow: 0 0 10px rgba(255, 75, 75, 0.3);
    }
    .download-button {
        background: linear-gradient(90deg, #00ddeb, #c084fc);
        color: white;
        border-radius: 10px;
        padding: 15px 30px;
        font-weight: bold;
        border: none;
        transition: all 0.4s ease;
        box-shadow: 0 0 10px rgba(0, 221, 235, 0.5);
        width: 100%;
    }
    .download-button:hover {
        background: linear-gradient(90deg, #00c4d4, #a855f7);
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(192, 132, 252, 0.7);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #b0b0b0;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #00ddeb;
        background: rgba(0, 221, 235, 0.2);
        box-shadow: 0 0 10px rgba(0, 221, 235, 0.5);
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #c084fc;
        background: rgba(192, 132, 252, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Load font Orbitron dari Google Fonts
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap" rel="stylesheet">', unsafe_allow_html=True)

def calculate_besson_rank(values):
    sorted_indices = np.argsort(-values)
    ranks = np.zeros_like(values, dtype=float)
    n = len(values)
    initial_ranks = np.arange(1, n + 1)
    sorted_values = values[sorted_indices]
    i = 0
    while i < n:
        current_value = sorted_values[i]
        tie_indices = np.where(sorted_values == current_value)[0]
        tie_count = len(tie_indices)
        mean_rank = np.mean(initial_ranks[tie_indices])
        for idx in tie_indices:
            ranks[sorted_indices[idx]] = mean_rank
        i += tie_count
    return ranks

def oreste_method_with_besson(alternatives, criteria, decision_matrix, criteria_ranks, criteria_weights, R):
    num_alts = len(alternatives)
    num_crits = len(criteria)
    besson_ranks = np.zeros_like(decision_matrix, dtype=float)
    for j in range(num_crits):
        besson_ranks[:, j] = calculate_besson_rank(decision_matrix[:, j])
    distance_scores_matrix = np.zeros_like(decision_matrix, dtype=float)
    for i in range(num_alts):
        for j in range(num_crits):
            r_a = besson_ranks[i, j]
            r_ref = criteria_ranks[j]
            distance_scores_matrix[i, j] = ((0.5 * (r_a ** R) + 0.5 * (r_ref ** R)) ** (1 / R))
    preference_values = np.zeros(num_alts)
    for i in range(num_alts):
        for j in range(num_crits):
            preference_values[i] += distance_scores_matrix[i, j] * criteria_weights[j]
    ranked_indices = np.argsort(preference_values)
    ranked_alternatives = [alternatives[i] for i in ranked_indices]
    return besson_ranks, distance_scores_matrix, np.sum(distance_scores_matrix, axis=1), preference_values, ranked_alternatives

# State untuk menyimpan hasil perhitungan
if 'results' not in st.session_state:
    st.session_state.results = None

# Sidebar untuk panduan dan info
with st.sidebar:
    st.header("Navigasi & Info")
    st.markdown("""
    **Langkah Penggunaan:**
    1. Masukkan data di tab **Input Data**.
    2. Klik **Hitung** untuk memproses.
    3. Lihat hasil di tab **Hasil Perhitungan**.
    4. Gunakan grafik interaktif untuk analisis.
    5. Unduh hasil dalam format CSV.

    **Tips:**
    - Pastikan total bobot kriteria mendekati 1.
    - Gunakan skor realistis (0-100).
    - Eksperimen dengan nilai R untuk hasil berbeda.
    """)
    st.markdown("---")
    st.markdown(f"**Dibuat oleh:** Ryanarynn")
    st.markdown(f"**Tanggal:** {datetime.now().strftime('%d %B %Y, %H:%M WIB')}")

# Header utama
st.markdown("<div class='card'><h1>Kalkulator ORESTE</h1></div>", unsafe_allow_html=True)
st.markdown("<div class='card'><p style='text-align: center; color: #b0b0b0;'>Metode ORESTE dengan Besson Rank untuk keputusan berbasis data.</p></div>", unsafe_allow_html=True)

# Tab untuk input dan hasil
tab1, tab2 = st.tabs(["📝 Input Data", "📊 Hasil Perhitungan"])

# Tab 1: Input Data
with tab1:
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("1️⃣ Tentukan Alternatif & Kriteria")
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            num_alternatives = st.number_input("Jumlah Alternatif", min_value=2, value=3, step=1, help="Minimal 2 alternatif.")
        with col2:
            num_criteria = st.number_input("Jumlah Kriteria", min_value=1, value=3, step=1, help="Minimal 1 kriteria.")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("2️⃣ Nama Alternatif & Kriteria")
        col3, col4 = st.columns([1, 1], gap="large")
        alternatives = []
        criteria = []
        with col3:
            st.markdown("**Alternatif**")
            for i in range(num_alternatives):
                alt = st.text_input(f"Alternatif {i+1}", value=f"A{i+1}", key=f"alt_{i}", help=f"Nama unik untuk alternatif {i+1}.")
                alternatives.append(alt)
        with col4:
            st.markdown("**Kriteria**")
            for j in range(num_criteria):
                crit = st.text_input(f"Kriteria {j+1}", value=f"K{j+1}", key=f"crit_{j}", help=f"Nama unik untuk kriteria {j+1}.")
                criteria.append(crit)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("3️⃣ Matriks Keputusan")
        st.markdown("Masukkan skor performa (0-100).")
        decision_matrix = np.zeros((num_alternatives, num_criteria))
        for i in range(num_alternatives):
            st.markdown(f"**{alternatives[i]}**")
            cols = st.columns(num_criteria)
            for j in range(num_criteria):
                with cols[j]:
                    decision_matrix[i, j] = st.number_input(
                        f"{criteria[j]}", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key=f"score_{i}_{j}",
                        help=f"Skor {alternatives[i]} untuk {criteria[j]}."
                    )
        st.markdown("</div>", unsafe_allow_html=True)

    # Generate criteria ranks otomatis
    criteria_ranks = list(range(1, num_criteria + 1))

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("4️⃣ Bobot Kriteria")
        st.markdown("Atur bobot (total = 1 idealnya).")
        criteria_weights = []
        cols = st.columns(num_criteria)
        for j in range(num_criteria):
            with cols[j]:
                weight = st.number_input(
                    f"Bobot {criteria[j]}", min_value=0.0, max_value=1.0, value=1.0/num_criteria, step=0.01, key=f"weight_{j}",
                    help=f"Bobot untuk {criteria[j]} (0-1)."
                )
                criteria_weights.append(weight)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("5️⃣ Koefisien R")
        st.markdown("Masukkan nilai R untuk perhitungan.")
        R = st.number_input("Nilai R", min_value=1.0, value=2.0, step=1.0, help="Nilai R (misalnya, 1, 2, 3).")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if st.button("Hitung Sekarang", use_container_width=True):
            if len(alternatives) != len(set(alternatives)) or len(criteria) != len(set(criteria)):
                st.error("Nama alternatif dan kriteria harus unik!")
            elif np.all(decision_matrix == 0):
                st.error("Matriks keputusan tidak boleh kosong atau berisi nol semua!")
            elif sum(criteria_weights) == 0:
                st.error("Total bobot kriteria tidak boleh nol!")
            elif R <= 0:
                st.error("Nilai R harus lebih besar dari 0!")
            else:
                with st.spinner("🔄 Sedang menghitung..."):
                    besson_ranks, distance_scores_matrix, total_distance_scores, preference_values, ranked_alternatives = oreste_method_with_besson(
                        alternatives, criteria, decision_matrix, criteria_ranks, criteria_weights, R
                    )
                    st.session_state.results = {
                        "besson_ranks": besson_ranks,
                        "distance_scores_matrix": distance_scores_matrix,
                        "total_distance_scores": total_distance_scores,
                        "preference_values": preference_values,
                        "ranked_alternatives": ranked_alternatives,
                        "alternatives": alternatives,
                        "criteria": criteria,
                        "decision_matrix": decision_matrix
                    }
                    st.success("✅ Perhitungan selesai! Lihat hasil di tab 'Hasil Perhitungan'.")
        st.markdown("</div>", unsafe_allow_html=True)

# Tab 2: Hasil Perhitungan
with tab2:
    if st.session_state.results:
        results = st.session_state.results
        besson_ranks = results["besson_ranks"]
        distance_scores_matrix = results["distance_scores_matrix"]
        total_distance_scores = results["total_distance_scores"]
        preference_values = results["preference_values"]
        ranked_alternatives = results["ranked_alternatives"]
        alternatives = results["alternatives"]
        criteria = results["criteria"]
        decision_matrix = results["decision_matrix"]

        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("📊 Hasil Perhitungan")
            
            st.markdown("**Matriks Keputusan (Skor Asli)**")
            st.dataframe(pd.DataFrame(decision_matrix, index=alternatives, columns=criteria), use_container_width=True)
            
            st.markdown("**Matriks Besson Rank**")
            st.dataframe(pd.DataFrame(besson_ranks, index=alternatives, columns=criteria), use_container_width=True)
            
            st.markdown("**Distance Score D(a, Cj)**")
            st.dataframe(pd.DataFrame(distance_scores_matrix, index=alternatives, columns=criteria), use_container_width=True)
            
            st.markdown("**Total Distance Score**")
            distance_df = pd.DataFrame(total_distance_scores, index=alternatives, columns=["Total Distance Score"])
            st.dataframe(distance_df, use_container_width=True)
            
            st.markdown("**Visualisasi Total Distance Score**")
            fig1 = px.bar(
                x=total_distance_scores,
                y=alternatives,
                orientation='h',
                labels={'x': 'Total Distance Score', 'y': 'Alternatif'},
                color=total_distance_scores,
                color_continuous_scale='Viridis',
                title="Total Distance Score per Alternatif"
            )
            fig1.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color="#e0e0e0",
                title_font_color="#c084fc",
                title_font_family="Orbitron"
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            st.markdown("**Nilai Preferensi (Vi)**")
            preference_df = pd.DataFrame(preference_values, index=alternatives, columns=["Nilai Preferensi (Vi)"])
            st.dataframe(preference_df, use_container_width=True)
            
            st.markdown("**Visualisasi Nilai Preferensi (Vi)**")
            fig2 = px.scatter(
                x=alternatives,
                y=preference_values,
                size=[30]*len(alternatives),
                color=preference_values,
                color_continuous_scale='Plasma',
                labels={'x': 'Alternatif', 'y': 'Nilai Preferensi (Vi)'},
                title="Nilai Preferensi (Vi) per Alternatif"
            )
            fig2.update_traces(marker=dict(line=dict(width=2, color='#00ddeb')))
            fig2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color="#e0e0e0",
                title_font_color="#c084fc",
                title_font_family="Orbitron"
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            st.markdown("**Ranking Alternatif**")
            ranked_df = pd.DataFrame(ranked_alternatives, columns=["Alternatif"], index=range(1, len(ranked_alternatives) + 1))
            st.dataframe(ranked_df, use_container_width=True)
            
            result_df = pd.DataFrame({
                "Alternatif": alternatives,
                "Total Distance Score": total_distance_scores,
                "Nilai Preferensi (Vi)": preference_values
            })
            result_df["Peringkat"] = result_df["Nilai Preferensi (Vi)"].rank(method="min").astype(int)
            result_df = result_df.sort_values("Peringkat")
            csv_buffer = io.StringIO()
            result_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Unduh Hasil (CSV)",
                data=csv_buffer.getvalue(),
                file_name="hasil_oreste_besson.csv",
                mime="text/csv",
                key="download_button",
                help="Unduh hasil dalam format CSV."
            )
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='card'><p style='color: #b0b0b0; text-align: center;'>Belum ada hasil. Silakan masukkan data dan klik 'Hitung Sekarang' di tab Input Data.</p></div>", unsafe_allow_html=True)