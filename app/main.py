import streamlit as st
import numpy as np
import pandas as pd
import io

def calculate_besson_rank(values):
    """
    Calculate Besson Rank for a list of values, handling ties by averaging ranks.
    Parameters:
        values: 1D array of performance scores
    Returns:
        ranks: 1D array of Besson ranks
    """
    # Sort values in descending order (highest to lowest) and get indices
    sorted_indices = np.argsort(-values)
    ranks = np.zeros_like(values, dtype=float)
    n = len(values)
    
    # Assign initial ranks (1 to n)
    initial_ranks = np.arange(1, n + 1)
    
    # Sort values and their corresponding initial ranks
    sorted_values = values[sorted_indices]
    
    # Calculate Besson ranks by handling ties
    i = 0
    while i < n:
        # Find all indices with the same value (tie)
        current_value = sorted_values[i]
        tie_indices = np.where(sorted_values == current_value)[0]
        tie_count = len(tie_indices)
        # Calculate mean rank for tied values
        mean_rank = np.mean(initial_ranks[tie_indices])
        # Assign mean rank to all tied values
        for idx in tie_indices:
            ranks[sorted_indices[idx]] = mean_rank
        i += tie_count
    
    return ranks

def oreste_method_with_besson(alternatives, criteria, decision_matrix, criteria_ranks, criteria_weights, R):
    """
    ORESTE Method with Besson Rank, Distance Score, and Preference Value (Vi)
    Parameters:
        alternatives: list of alternative names
        criteria: list of criteria names
        decision_matrix: 2D array of performance scores (rows: alternatives, cols: criteria)
        criteria_ranks: list of ranks for criteria (r_CjR, used as r_ref)
        criteria_weights: list of weights for criteria (Wj)
        R: Coefficient value for exponentiation
    Returns:
        besson_ranks: matrix of Besson ranks per criterion
        distance_scores_matrix: matrix of Distance Scores D(a, Cj) per alternative and criterion
        total_distance_scores: total Distance Scores per alternative
        preference_values: preference values (Vi) for each alternative
        ranked_alternatives: list of alternatives sorted by preference value
    """
    num_alts = len(alternatives)
    num_crits = len(criteria)

    # Step 1: Convert decision matrix to Besson ranks per criterion
    besson_ranks = np.zeros_like(decision_matrix, dtype=float)
    for j in range(num_crits):
        besson_ranks[:, j] = calculate_besson_rank(decision_matrix[:, j])

    # Step 2: Calculate Distance Scores D(a, Cj) for each alternative and criterion
    # D(a_j, C_j) = [ (1/2) * (r_(a))^R + (1/2) * (r_ref)^R ]^(1/R)
    # r_ref is the rank of the criterion (criteria_ranks[j])
    distance_scores_matrix = np.zeros_like(decision_matrix, dtype=float)
    for i in range(num_alts):
        for j in range(num_crits):
            r_a = besson_ranks[i, j]  # Besson rank of alternative a on criterion Cj
            r_ref = criteria_ranks[j]  # Rank of criterion Cj as r_ref
            # Apply the formula
            distance_scores_matrix[i, j] = ((0.5 * (r_a ** R) + 0.5 * (r_ref ** R)) ** (1 / R))

    # Step 3: Calculate Preference Values (Vi) = sum(D_j * W_j) for each alternative
    preference_values = np.zeros(num_alts)
    for i in range(num_alts):
        for j in range(num_crits):
            preference_values[i] += distance_scores_matrix[i, j] * criteria_weights[j]

    # Step 4: Rank alternatives based on Preference Values (lower is better)
    ranked_indices = np.argsort(preference_values)
    ranked_alternatives = [alternatives[i] for i in ranked_indices]

    return besson_ranks, distance_scores_matrix, np.sum(distance_scores_matrix, axis=1), preference_values, ranked_alternatives

# Streamlit App
st.set_page_config(page_title="Kalkulator ORESTE", layout="wide")
st.title("Kalkulator Sistem Pendukung Keputusan ORESTE (Metode Besson)")
st.markdown("""
Kalkulator ini menggunakan metode ORESTE dengan pendekatan Besson Rank, Distance Score, dan Nilai Preferensi (Vi).
Masukkan jumlah alternatif, kriteria, skor performa, peringkat kriteria, bobot kriteria, dan koefisien R untuk mendapatkan hasil.
""")

# Input for number of alternatives and criteria
st.subheader("Masukkan Alternatif dan Kriteria")
col1, col2 = st.columns(2)
with col1:
    num_alternatives = st.number_input("Jumlah Alternatif", min_value=2, value=3, step=1)
with col2:
    num_criteria = st.number_input("Jumlah Kriteria", min_value=1, value=3, step=1)

# Input for alternative and criteria names
st.subheader("Nama Alternatif dan Kriteria")
alternatives = []
criteria = []
col3, col4 = st.columns(2)
with col3:
    st.write("Alternatif")
    for i in range(num_alternatives):
        alt = st.text_input(f"Alternatif {i+1}", value=f"A{i+1}", key=f"alt_{i}")
        alternatives.append(alt)
with col4:
    st.write("Kriteria")
    for j in range(num_criteria):
        crit = st.text_input(f"Kriteria {j+1}", value=f"K{j+1}", key=f"crit_{j}")
        criteria.append(crit)

# Input for decision matrix
st.subheader("Matriks Keputusan (Skor Performa)")
st.markdown("Masukkan skor performa untuk setiap alternatif pada setiap kriteria (misalnya, 0-100).")
decision_matrix = np.zeros((num_alternatives, num_criteria))
for i in range(num_alternatives):
    st.write(f"Skor untuk {alternatives[i]}")
    cols = st.columns(num_criteria)
    for j in range(num_criteria):
        with cols[j]:
            decision_matrix[i, j] = st.number_input(
                f"{criteria[j]}", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key=f"score_{i}_{j}"
            )

# Input for criteria ranks (r_CjR)
st.subheader("Peringkat Kriteria (r_CjR)")
st.markdown("Masukkan peringkat untuk setiap kriteria (1 = paling penting, digunakan sebagai r_ref).")
criteria_ranks = []
cols = st.columns(num_criteria)
for j in range(num_criteria):
    with cols[j]:
        rank = st.number_input(
            f"Peringkat {criteria[j]}", min_value=1, value=j+1, step=1, key=f"rank_{j}"
        )
        criteria_ranks.append(rank)

# Input for criteria weights (Wj)
st.subheader("Bobot Kriteria (Wj)")
st.markdown("Masukkan bobot untuk setiap kriteria (misalnya, 0-1, total bobot sebaiknya 1).")
criteria_weights = []
cols = st.columns(num_criteria)
for j in range(num_criteria):
    with cols[j]:
        weight = st.number_input(
            f"Bobot {criteria[j]}", min_value=0.0, max_value=1.0, value=0.1, step=0.01, key=f"weight_{j}"
        )
        criteria_weights.append(weight)

# Input for coefficient R
st.subheader("Koefisien R (Nilai Ketetapan Perpangkatan)")
st.markdown("Masukkan nilai koefisien R (misalnya, 1, 2, 3, atau lainnya).")
R = st.number_input("Nilai R", min_value=1.0, value=1.0, step=1.0)

# Validation and calculation
if st.button("Hitung"):
    # Check for valid inputs
    if len(alternatives) != len(set(alternatives)) or len(criteria) != len(set(criteria)):
        st.error("Nama alternatif dan kriteria harus unik!")
    elif np.all(decision_matrix == 0):
        st.error("Matriks keputusan tidak boleh kosong atau berisi nol semua!")
    elif len(set(criteria_ranks)) != len(criteria_ranks):
        st.error("Peringkat kriteria harus unik!")
    elif sum(criteria_weights) == 0:
        st.error("Total bobot kriteria tidak boleh nol!")
    elif R <= 0:
        st.error("Nilai R harus lebih besar dari 0!")
    else:
        # Run ORESTE method with Besson Rank
        besson_ranks, distance_scores_matrix, total_distance_scores, preference_values, ranked_alternatives = oreste_method_with_besson(
            alternatives, criteria, decision_matrix, criteria_ranks, criteria_weights, R
        )
        
        # Display results
        st.subheader("Hasil Perhitungan")
        
        st.write("**Matriks Keputusan (Skor Asli)**")
        st.dataframe(pd.DataFrame(decision_matrix, index=alternatives, columns=criteria))
        
        st.write("**Matriks Besson Rank (Per Kriteria)**")
        st.dataframe(pd.DataFrame(besson_ranks, index=alternatives, columns=criteria))
        
        st.write("**Distance Score D(a, Cj) per Kriteria**")
        st.dataframe(pd.DataFrame(distance_scores_matrix, index=alternatives, columns=criteria))
        
        st.write("**Total Distance Score per Alternatif**")
        distance_df = pd.DataFrame(total_distance_scores, index=alternatives, columns=["Total Distance Score"])
        st.dataframe(distance_df)
        
        st.write("**Nilai Preferensi (Vi)**")
        preference_df = pd.DataFrame(preference_values, index=alternatives, columns=["Nilai Preferensi (Vi)"])
        st.dataframe(preference_df)
        
        st.write("**Urutan Alternatif**")
        ranked_df = pd.DataFrame(ranked_alternatives, columns=["Alternatif"], index=range(1, len(ranked_alternatives) + 1))
        st.dataframe(ranked_df)
        
        # Download results as CSV
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
            label="Unduh Hasil sebagai CSV",
            data=csv_buffer.getvalue(),
            file_name="hasil_oreste_besson.csv",
            mime="text/csv"
        )

# Instructions
st.markdown("""
### Cara Penggunaan
1. Tentukan jumlah alternatif dan kriteria.
2. Masukkan nama unik untuk setiap alternatif dan kriteria.
3. Isi skor performa untuk setiap alternatif pada setiap kriteria (misalnya, 0-100).
4. Berikan peringkat unik untuk setiap kriteria (1 = paling penting, digunakan sebagai r_ref).
5. Berikan bobot untuk setiap kriteria (misalnya, 0-1, total bobot sebaiknya 1).
6. Masukkan nilai koefisien R (misalnya, 1, 2, 3, atau lainnya).
7. Klik tombol "Hitung" untuk melihat hasil.
8. Unduh hasil dalam format CSV jika diperlukan.
""")