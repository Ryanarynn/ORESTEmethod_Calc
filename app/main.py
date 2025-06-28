import streamlit as st
import numpy as np
import pandas as pd
import io
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
import openpyxl
import xlrd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import tempfile

#############################################
# START OF CSS AND STYLING CONFIGURATION
#############################################

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

#############################################
# END OF CSS AND STYLING CONFIGURATION
#############################################

#############################################
# START OF CORE FUNCTIONS AND LOGIC
#############################################

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

def validate_inputs(decision_matrix, criteria_weights, alternatives, criteria):
    """Validate all input data"""
    errors = []
    
    # Check for negative values
    if (decision_matrix < 0).any():
        errors.append("Matriks keputusan tidak boleh mengandung nilai negatif")
    
    # Check matrix size
    if decision_matrix.size > 1000:  # arbitrary limit
        errors.append("Ukuran matriks terlalu besar")
    
    # MODIFIED: Check weight sum within a range
    if not (0.999 <= sum(criteria_weights) <= 1.2):
        errors.append("Total bobot harus di antara 0.999 dan 1.2")
    
    return errors

def validate_imported_dataframe(df):
    """Validate imported dataframe format"""
    try:
        # Check required columns
        if 'Alternatif' not in df.columns:
            return False, "Kolom 'Alternatif' tidak ditemukan"
        
        # Check if there are criteria columns
        if len(df.columns) < 2:
            return False, "File harus memiliki minimal 1 kriteria"
            
        # Check for empty cells
        if df.isnull().values.any():
            return False, "Data tidak boleh kosong"
            
        # Check numeric values
        criteria_cols = df.columns[1:]
        for col in criteria_cols:
            if not pd.to_numeric(df[col], errors='coerce').notnull().all():
                return False, f"Kolom {col} harus berisi angka"
            if not all(0 <= x <= 100 for x in df[col]):
                return False, f"Nilai dalam kolom {col} harus antara 0-100"
                
        return True, "Data valid"
    except Exception as e:
        return False, f"Error validasi: {str(e)}"

def import_from_excel(file):
    """Import data from Excel/CSV file with enhanced error handling"""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file, engine='openpyxl')
        
        # Validate imported data
        is_valid, message = validate_imported_dataframe(df)
        if not is_valid:
            st.error(message)
            return None
            
        # Clean and process data
        df = df.fillna(0)  # Replace any remaining NaN with 0
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df
    except Exception as e:
        st.error(f"Error saat import file: {str(e)}")
        return None

def plot_radar_chart(decision_matrix, alternatives, criteria):
    """Create radar chart for alternatives comparison"""
    fig = go.Figure()
    for i, alt in enumerate(alternatives):
        fig.add_trace(go.Scatterpolar(
            r=decision_matrix[i],
            theta=criteria,
            name=alt
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
    return fig

def export_to_csv(results):
    """Export complete results to CSV"""
    # Create DataFrames for each step
    decision_df = pd.DataFrame(results["decision_matrix"], 
                             index=results["alternatives"],
                             columns=results["criteria"])
    decision_df.index.name = "Alternatif"
    
    besson_df = pd.DataFrame(results["besson_ranks"],
                           index=results["alternatives"],
                           columns=results["criteria"])
    besson_df.index.name = "Alternatif"
    
    distance_df = pd.DataFrame(results["distance_scores_matrix"],
                             index=results["alternatives"],
                             columns=results["criteria"])
    distance_df.index.name = "Alternatif"
    
    final_df = pd.DataFrame({
        "Alternatif": results["alternatives"],
        "Total Distance Score": results["total_distance_scores"],
        "Nilai Preferensi": results["preference_values"],
        "Peringkat": pd.Series(results["preference_values"]).rank(method="min").astype(int)
    }).set_index("Alternatif")
    
    # Combine all into one string buffer
    output = io.StringIO()
    
    output.write("HASIL PERHITUNGAN METODE ORESTE\n\n")
    
    output.write("1. MATRIKS KEPUTUSAN\n")
    decision_df.to_csv(output)
    output.write("\n")
    
    output.write("2. MATRIKS BESSON RANK\n")
    besson_df.to_csv(output)
    output.write("\n")
    
    output.write("3. DISTANCE SCORE MATRIX\n")
    distance_df.to_csv(output)
    output.write("\n")
    
    output.write("4. HASIL AKHIR\n")
    final_df.to_csv(output)
    
    return output.getvalue()

def export_to_pdf(results):
    """Export complete results to PDF"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            doc = SimpleDocTemplate(
                tmp_file.name,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=1
            )
            story.append(Paragraph("Hasil Perhitungan ORESTE", title_style))
            story.append(Spacer(1, 20))
            
            # 1. Matriks Keputusan
            story.append(Paragraph("1. Matriks Keputusan", styles['Heading2']))
            decision_data = [[""]+results["criteria"]]
            for i, alt in enumerate(results["alternatives"]):
                row = [alt] + list(results["decision_matrix"][i])
                decision_data.append(row)
            
            t = Table(decision_data)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(t)
            story.append(Spacer(1, 20))
            
            # 2. Matriks Besson Rank
            story.append(Paragraph("2. Matriks Besson Rank", styles['Heading2']))
            besson_data = [[""]+results["criteria"]]
            for i, alt in enumerate(results["alternatives"]):
                row = [alt] + [f"{x:.2f}" for x in results["besson_ranks"][i]]
                besson_data.append(row)
            
            t = Table(besson_data)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(t)
            story.append(Spacer(1, 20))
            
            # 3. Distance Score Matrix
            story.append(Paragraph("3. Distance Score Matrix", styles['Heading2']))
            distance_data = [[""]+results["criteria"]]
            for i, alt in enumerate(results["alternatives"]):
                row = [alt] + [f"{x:.2f}" for x in results["distance_scores_matrix"][i]]
                distance_data.append(row)
            
            t = Table(distance_data)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(t)
            story.append(Spacer(1, 20))
            
            # 4. Hasil Akhir
            story.append(Paragraph("4. Hasil Akhir", styles['Heading2']))
            
            # Sort by preference values
            sorted_indices = np.argsort(results["preference_values"])
            final_data = [["Peringkat", "Alternatif", "Total Distance", "Nilai Preferensi"]]
            
            for rank, idx in enumerate(sorted_indices, 1):
                final_data.append([
                    str(rank),
                    results["alternatives"][idx],
                    f"{results['total_distance_scores'][idx]:.2f}",
                    f"{results['preference_values'][idx]:.4f}"
                ])
            
            t = Table(final_data)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(t)
            
            # Build PDF
            doc.build(story)
            
            with open(tmp_file.name, 'rb') as pdf_file:
                return pdf_file.read()
                
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None

def get_template_data():
    """Create template Excel file"""
    df_template = pd.DataFrame({
        'Alternatif': ['A1', 'A2', 'A3'],
        'K1': [0, 0, 0],
        'K2': [0, 0, 0],
        'K3': [0, 0, 0]
    })
    return df_template

def create_download_link(df):
    """Generate download link for template"""
    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False, engine='openpyxl')
    excel_buffer.seek(0)
    b64 = base64.b64encode(excel_buffer.read()).decode()
    return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="template_oreste.xlsx">Download Template Excel</a>'

def parse_imported_data(df):
    """Parse imported data into calculator format"""
    try:
        alternatives = df['Alternatif'].tolist()
        criteria = df.columns[1:].tolist()
        decision_matrix = df.iloc[:, 1:].to_numpy()
        return alternatives, criteria, decision_matrix
    except Exception as e:
        st.error(f"Format data tidak sesuai: {str(e)}")
        return None, None, None

# Tambahkan fungsi untuk mengupdate form
def update_form_with_imported_data(imported_data):
    """Update form fields with imported data"""
    if imported_data:
        st.session_state.num_alternatives = len(imported_data["alternatives"])
        st.session_state.num_criteria = len(imported_data["criteria"])
        
        # Update alternatif names
        for i, alt in enumerate(imported_data["alternatives"]):
            st.session_state[f"alt_{i}"] = alt
            
        # Update kriteria names
        for j, crit in enumerate(imported_data["criteria"]):
            st.session_state[f"crit_{j}"] = crit
            
        # Update matrix values
        for i in range(len(imported_data["alternatives"])):
            for j in range(len(imported_data["criteria"])):
                st.session_state[f"score_{i}_{j}"] = imported_data["decision_matrix"][i][j]

def generate_detailed_calculations(alternatives, criteria, decision_matrix, criteria_ranks, criteria_weights, R, besson_ranks, distance_scores_matrix, preference_values):
    """Generate detailed step-by-step calculations"""
    steps = []
    
    # Step 1: Besson Rank Calculation
    steps.append({
        'title': "1️⃣ Perhitungan Besson Rank",
        'description': """
        Besson Rank dihitung untuk setiap kriteria dengan langkah:
        1. Urutkan nilai dari besar ke kecil
        2. Jika ada nilai yang sama, gunakan rata-rata peringkat
        """,
        'details': []
    })
    
    for j, criterion in enumerate(criteria):
        step_detail = f"Kriteria: {criterion}\n"
        values = decision_matrix[:, j]
        sorted_indices = np.argsort(-values)
        sorted_values = values[sorted_indices]
        step_detail += "Nilai terurut: " + ", ".join([f"{alternatives[i]}({sorted_values[idx]:.2f})" 
                                                     for idx, i in enumerate(sorted_indices)])
        steps[-1]['details'].append(step_detail)
    
    # Step 2: Distance Score Calculation
    steps.append({
        'title': "2️⃣ Perhitungan Distance Score",
        'description': f"""
        Distance Score dihitung dengan rumus:
        D(a,Cj) = (0.5 * (r_a^R) + 0.5 * (r_ref^R))^(1/R)
        dimana:
        - r_a = Besson rank alternatif
        - r_ref = Peringkat kriteria
        - R = {R} (nilai yang digunakan)
        """,
        'details': []
    })
    
    for i, alt in enumerate(alternatives):
        for j, crit in enumerate(criteria):
            r_a = besson_ranks[i, j]
            r_ref = criteria_ranks[j]
            d_score = distance_scores_matrix[i, j]
            calc = f"{alt} untuk {crit}:\n"
            calc += f"D = (0.5 * ({r_a:.2f}^{R}) + 0.5 * ({r_ref}^{R}))^(1/{R})\n"
            calc += f"D = {d_score:.4f}"
            steps[-1]['details'].append(calc)
    
    # Step 3: Preference Value Calculation
    steps.append({
        'title': "3️⃣ Perhitungan Nilai Preferensi",
        'description': """
        Nilai preferensi dihitung dengan menjumlahkan hasil perkalian
        distance score dengan bobot kriteria:
        Vi = Σ(D(a,Cj) * Wj)
        """,
        'details': []
    })
    
    for i, alt in enumerate(alternatives):
        calc = f"Alternatif {alt}:\n"
        terms = []
        for j, crit in enumerate(criteria):
            d_score = distance_scores_matrix[i, j]
            weight = criteria_weights[j]
            term = f"({d_score:.4f} × {weight:.3f})"
            terms.append(term)
        calc += "Vi = " + " + ".join(terms) + f" = {preference_values[i]:.4f}"
        steps[-1]['details'].append(calc)
    
    return steps

def reset_calculation():
    """Reset all calculation data and form inputs"""
    keys_to_delete = []
    
    # Find all keys to delete
    for key in st.session_state:
        # Keep widget keys but reset their values through the widgets themselves
        if not key.startswith(('num_', 'alt_', 'crit_', 'score_', 'weight_')):
            keys_to_delete.append(key)
    
    # Delete non-widget keys
    for key in keys_to_delete:
        del st.session_state[key]

#############################################
# END OF CORE FUNCTIONS AND LOGIC
#############################################

#############################################
# START OF APP LAYOUT AND MAIN LOGIC
#############################################

# State untuk menyimpan hasil perhitungan
if 'results' not in st.session_state:
    st.session_state.results = None

# Sidebar untuk panduan dan info
with st.sidebar:
    st.header("ORESTE Calculator")
    
    # Create tabs in sidebar
    sidebar_tabs = st.tabs(["📚 Panduan", "ℹ️ Info", "📥 Import"])
    
    # Tab Panduan
    with sidebar_tabs[0]:
        st.markdown("""
        ### Langkah Penggunaan
        1. Masukkan data di tab **Input Data**
        2. Klik **Hitung** untuk memproses
        3. Lihat hasil di tab **Hasil Perhitungan**
        4. Gunakan grafik interaktif untuk analisis
        5. Unduh hasil dalam format CSV/PDF

        ### Tips Penggunaan
        - Pastikan total bobot kriteria berada di rentang 0.999 - 1.2
        - Gunakan skor 0-100
        - Eksperimen dengan nilai R berbeda
        """)
    
    # Tab Info
    with sidebar_tabs[1]:
        st.markdown("### Tentang Aplikasi")
        st.markdown(f"**Dibuat oleh:** Ryanarynn")
        st.markdown(f"**Tanggal:** {datetime.now().strftime('%d %B %Y')}")
        st.markdown(f"**Waktu:** {datetime.now().strftime('%H:%M WIB')}")
        
        with st.expander("❓ Apa itu ORESTE?"):
            st.markdown("""
            ORESTE adalah metode pengambilan keputusan multi-kriteria
            yang menggunakan Besson Rank untuk mengurutkan alternatif
            berdasarkan kriteria yang ditentukan.
            """)
    
    # Tab Import
    with sidebar_tabs[2]:
        st.markdown("### Import Data")
        
        with st.expander("ℹ️ Format Data"):
            st.info("""
            **Format yang Didukung:**
            - Excel (.xlsx)
            - CSV (.csv)
            
            **Struktur Data:**
            - Kolom 1: 'Alternatif'
            - Kolom 2+: Nilai kriteria (0-100)
            - Tidak boleh ada sel kosong
            """)
        
        # Template download
        template_df = get_template_data()
        st.markdown("### Template")
        st.markdown(create_download_link(template_df), unsafe_allow_html=True)
        
        st.markdown("### Upload File")
        uploaded_file = st.file_uploader("Pilih file Excel/CSV", type=['xlsx', 'csv'])
        
        if uploaded_file:
            with st.spinner("Memproses file..."):
                imported_df = import_from_excel(uploaded_file)
                if imported_df is not None:
                    try:
                        alternatives = imported_df['Alternatif'].tolist()
                        criteria = imported_df.columns[1:].tolist()
                        decision_matrix = imported_df.iloc[:, 1:].to_numpy()
                        
                        st.session_state.imported_data = {
                            "alternatives": alternatives,
                            "criteria": criteria,
                            "decision_matrix": decision_matrix
                        }
                        
                        update_form_with_imported_data(st.session_state.imported_data)
                        
                        st.success("✅ Data berhasil diimport!")
                        
                        with st.expander("🔍 Preview Data"):
                            st.dataframe(imported_df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

# Header utama
st.markdown("<div class='card'><h1>Kalkulator ORESTE</h1></div>", unsafe_allow_html=True)
st.markdown("<div class='card'><p style='text-align: center; color: #b0b0b0;'>Metode ORESTE dengan Besson Rank untuk keputusan berbasis data.</p></div>", unsafe_allow_html=True)

# Tab untuk input dan hasil
tab1, tab2 = st.tabs(["📝 Input Data", "📊 Hasil Perhitungan"])

# Tab 1: Input Data
with tab1:
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("1. Tentukan Alternatif & Kriteria")
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            num_alternatives = st.number_input("Jumlah Alternatif", 
                                            min_value=2, 
                                            value=st.session_state.get('num_alternatives', 3),
                                            step=1,
                                            help="Minimal 2 alternatif.",
                                            key="num_alternatives")
        with col2:
            num_criteria = st.number_input("Jumlah Kriteria",
                                         min_value=1,
                                         value=st.session_state.get('num_criteria', 3),
                                         step=1,
                                         help="Minimal 1 kriteria.",
                                         key="num_criteria")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("2. Nama Alternatif & Kriteria")
        col3, col4 = st.columns([1, 1], gap="large")
        alternatives = []
        criteria = []
        with col3:
            st.markdown("**Alternatif**")
            for i in range(num_alternatives):
                alt = st.text_input(f"Alternatif {i+1}",
                                  value=st.session_state.get(f"alt_{i}", f"A{i+1}"),
                                  key=f"alt_{i}",
                                  help=f"Nama unik untuk alternatif {i+1}.")
                alternatives.append(alt)
        with col4:
            st.markdown("**Kriteria**")
            for j in range(num_criteria):
                crit = st.text_input(f"Kriteria {j+1}",
                                   value=st.session_state.get(f"crit_{j}", f"K{j+1}"),
                                   key=f"crit_{j}",
                                   help=f"Nama unik untuk kriteria {j+1}.")
                criteria.append(crit)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("3. Matriks Keputusan")
        st.markdown("Masukkan skor performa (0-100).")
        decision_matrix = np.zeros((num_alternatives, num_criteria))
        for i in range(num_alternatives):
            st.markdown(f"**{alternatives[i]}**")
            cols = st.columns(num_criteria)
            for j in range(num_criteria):
                with cols[j]:
                    decision_matrix[i, j] = st.number_input(
                        f"{criteria[j]}",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(st.session_state.get(f"score_{i}_{j}", 0.0)),
                        step=0.1,
                        key=f"score_{i}_{j}",
                        help=f"Skor {alternatives[i]} untuk {criteria[j]}."
                    )
        st.markdown("</div>", unsafe_allow_html=True)

    # Generate criteria ranks otomatis
    criteria_ranks = list(range(1, num_criteria + 1))

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("4. Bobot Kriteria")
        st.markdown("Atur bobot (total disarankan 0.999 - 1.2).")
        criteria_weights = []
        cols = st.columns(num_criteria)
        for j in range(num_criteria):
            with cols[j]:
                weight = st.number_input(
                f"Bobot {criteria[j]}", min_value=0.0, max_value=1.0, value=1.0/num_criteria, step=0.01, key=f"weight_{j}",
                help=f"Bobot untuk {criteria[j]} (0-1)."
            )
            criteria_weights.append(weight)
        
        # MODIFIED: Total bobot kriteria
        total_weight = sum(criteria_weights)
        if 0.999 <= total_weight <= 1.2:
            weight_status = "✅ Total bobot berada dalam rentang yang diterima!"
            color = "#00ddeb"
        else:
            weight_status = "⚠️ Total bobot di luar rentang yang disarankan!"
            color = "#ff4b4b"
        st.markdown(f"<p style='color: {color}; text-align: center;'>{weight_status} <b>Total: {total_weight:.3f}</b></p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("5. Koefisien R")
        st.markdown("Masukkan nilai R untuk perhitungan.")
        R = st.number_input("Nilai R", min_value=1.0, value=2.0, step=1.0, help="Nilai R (misalnya, 1, 2, 3).")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Reset Input", use_container_width=True):
                reset_calculation()
                st.rerun()
                
        with col2:
            if st.button("Hitung", use_container_width=True):
                errors = validate_inputs(decision_matrix, criteria_weights, alternatives, criteria)
                if errors:
                    for error in errors:
                        st.error(error)
                else:
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
            
            st.markdown("**Radar Chart Perbandingan**")
            radar_fig = plot_radar_chart(decision_matrix, alternatives, criteria)
            st.plotly_chart(radar_fig, use_container_width=True)
            
            st.markdown("**Ranking Alternatif**")
            ranked_df = pd.DataFrame(ranked_alternatives, columns=["Alternatif"], index=range(1, len(ranked_alternatives) + 1))
            st.dataframe(ranked_df, use_container_width=True)
            
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("📝 Perhitungan Detail")
            
            detailed_steps = generate_detailed_calculations(
                alternatives, criteria, decision_matrix, 
                criteria_ranks, criteria_weights, R,
                besson_ranks, distance_scores_matrix, preference_values
            )
            
            for step in detailed_steps:
                with st.expander(step['title']):
                    st.markdown(step['description'])
                    st.markdown("**Detail Perhitungan:**")
                    for detail in step['details']:
                        st.markdown(f"```\n{detail}\n```")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            csv_data = export_to_csv(st.session_state.results)
            # Create columns without any gap parameter
            col1, col2 = st.columns([1, 1])  # Removed gap parameter
            with col1:
                st.download_button(
                    label="📥 Unduh Hasil (CSV)",
                    data=csv_data,
                    file_name="hasil_oreste_besson.csv",
                    mime="text/csv",
                    key="download_button",
                    help="Unduh hasil lengkap dalam format CSV.",
                    use_container_width=True
                )
            with col2:
                pdf_data = export_to_pdf(st.session_state.results)
                if pdf_data:
                    st.download_button(
                        label="📑 Unduh Hasil (PDF)",
                        data=pdf_data,
                        file_name="hasil_oreste_besson.pdf",
                        mime="application/pdf",
                        help="Unduh hasil lengkap dalam format PDF.",
                        use_container_width=True
                    )
                else:
                    st.error("Gagal membuat PDF")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            if st.button("🔄 Hitung Ulang", use_container_width=True):
                reset_calculation()
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='card'><p style='color: #b0b0b0; text-align: center;'>Belum ada hasil. Silakan masukkan data dan klik 'Hitung Sekarang' di tab Input Data.</p></div>", unsafe_allow_html=True)

#############################################
# END OF APP LAYOUT AND MAIN LOGIC
#############################################