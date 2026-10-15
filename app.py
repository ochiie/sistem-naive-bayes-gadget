"""
app.py
===============================================================================
Sistem Analisis Kecanduan Gadget pada Siswa — Metode Gaussian Naive Bayes
(diimplementasikan manual/from scratch, tampilan formal).

STRUKTUR PROYEK SEDERHANA (2 file utama):
    - app.py   -> file ini. Berisi seluruh TAMPILAN & alur setiap menu/halaman.
    - core.py  -> berisi seluruh LOGIKA (tema, pembaca Excel, preprocessing,
                  algoritma Naive Bayes manual, pipeline analisis, laporan).

Setiap menu (dulu berupa file terpisah di folder pages/) sekarang menjadi
1 fungsi `halaman_...()` di bawah, dan dipilih lewat menu di sidebar kiri.
Tidak ada fitur yang dihilangkan — hanya jumlah filenya yang disederhanakan.
===============================================================================
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

import core
import database

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Password Admin — GANTI sesuai kebutuhan sebelum sistem dipakai sungguhan.
# ---------------------------------------------------------------------------
ADMIN_PASSWORD = "admin123"

st.set_page_config(
    page_title="Sistem Analisis Kecanduan Gadget",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Inisialisasi session_state (agar semua halaman aman diakses dalam urutan apa pun)
# ---------------------------------------------------------------------------
DEFAULT_STATE = {
    "df_raw": None,
    "df_clean": None,
    "df_clean_asli": None,
    "feature_cols": None,
    "target_col": None,
    "label_encoders": None,
    "target_encoder": None,
    "X_train": None, "X_test": None, "y_train": None, "y_test": None,
    "train_idx": None, "test_idx": None,
    "model": None,
    "y_pred": None, "y_proba": None,
    "metrics": None, "confusion_matrix": None,
    "hasil_klasifikasi_df": None,
    "hasil_seluruh_data": None,
    "ringkasan_kelas": None,
    "activity_log": None,
    "kuesioner_entries": None,
    "peran": None,
    "admin_terautentikasi": False,
    "versi_form": 0,
    "hasil_kuesioner_terakhir": None,
}
for key, val in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = val

core.apply_global_style()


def label_terbaca(arr_or_val, target_encoder):
    """Bantu ubah kode angka kelas -> nama label asli (Rendah/Sedang/Tinggi)."""
    if target_encoder is None:
        return arr_or_val
    if np.ndim(arr_or_val) == 0:
        return target_encoder.inverse_transform([arr_or_val])[0]
    return target_encoder.inverse_transform(arr_or_val)


def reset_dari_upload():
    """Reset semua tahap turunan saat dataset baru diunggah."""
    for key in ["df_clean", "df_clean_asli", "feature_cols", "target_col",
                "label_encoders", "target_encoder", "X_train", "X_test",
                "y_train", "y_test", "train_idx", "test_idx", "model",
                "y_pred", "y_proba", "metrics", "confusion_matrix",
                "hasil_klasifikasi_df", "hasil_seluruh_data", "ringkasan_kelas"]:
        st.session_state[key] = None


def reset_dari_preprocessing():
    for key in ["X_train", "X_test", "y_train", "y_test", "train_idx", "test_idx",
                "model", "y_pred", "y_proba", "metrics", "hasil_klasifikasi_df"]:
        st.session_state[key] = None


def reset_dari_split():
    for key in ["model", "y_pred", "y_proba", "metrics", "hasil_klasifikasi_df"]:
        st.session_state[key] = None


def reset_dari_training():
    for key in ["y_pred", "y_proba", "metrics", "hasil_klasifikasi_df"]:
        st.session_state[key] = None


# ===============================================================================
# KONFIGURASI KUESIONER (pertanyaan, kolom, & skala penilaian)
# ===============================================================================

# ===============================================================================
# KONFIGURASI KUESIONER — sesuai Instrumen Penelitian asli
# "Kecanduan Gadget pada Siswa SMP Kelas VII, VIII, dan IX" (15 pernyataan, skala 1-5)
# ===============================================================================

KOLOM_IDENTITAS_KUESIONER = ["Nama", "Kelas", "Jenis Kelamin", "Usia"]
KOLOM_FITUR_KUESIONER = [f"P{i}" for i in range(1, 16)]
KOLOM_STANDAR_DATASET = (
    ["No"] + KOLOM_IDENTITAS_KUESIONER + KOLOM_FITUR_KUESIONER + ["Total Skor", "Tingkat_Kecanduan"]
)

DAFTAR_PERNYATAAN_KUESIONER = [
    "Saya menggunakan gadget lebih dari 4 jam dalam satu hari.",
    "Saya sering bermain gadget sebelum berangkat ke sekolah.",
    "Saya lebih sering menggunakan gadget untuk hiburan daripada untuk belajar.",
    "Saya sering menunda mengerjakan tugas sekolah karena bermain gadget.",
    "Saya sering menggunakan gadget ketika guru sedang menjelaskan pelajaran.",
    "Saya sering menggunakan gadget saat jam pelajaran berlangsung.",
    "Saya merasa penggunaan gadget membuat saya sulit berkonsentrasi saat belajar.",
    "Saya merasa penggunaan gadget memengaruhi prestasi belajar saya di sekolah.",
    "Saya sering membuka media sosial atau bermain game saat berada di lingkungan sekolah.",
    "Saya lebih memilih menggunakan gadget daripada berinteraksi dengan guru atau teman.",
    "Saya sering mengabaikan kegiatan atau hobi lain karena lebih memilih menggunakan gadget.",
    "Saya tetap menggunakan gadget meskipun mata saya terasa lelah.",
    "Saya merasa sulit mengurangi waktu penggunaan gadget setiap hari.",
    "Saya merasa hari saya terasa kurang lengkap jika tidak menggunakan gadget.",
    "Saya merasa penggunaan gadget memengaruhi aktivitas sehari-hari saya.",
]

# Pilihan jawaban ditampilkan sebagai huruf (A-E); nilai numerik 1-5 hanya dipakai
# untuk perhitungan Total Skor dan analisis Naive Bayes.
OPSI_JAWABAN = ["A", "B", "C", "D", "E"]
KETERANGAN_JAWABAN = [
    ("A", "Sangat Tidak Setuju (STS)"),
    ("B", "Tidak Setuju (TS)"),
    ("C", "Netral (N)"),
    ("D", "Setuju (S)"),
    ("E", "Sangat Setuju (SS)"),
]
SKOR_JAWABAN = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
TEKS_JAWABAN = {h: ket.split(" (")[0] for h, ket in KETERANGAN_JAWABAN}


def hitung_klasifikasi(new_row: dict):
    """
    Mengambil 1 jawaban kuesioner (dict), lalu menghitungnya memakai model Naive Bayes
    yang sedang aktif di session_state (jika sudah dilatih). Mengembalikan None bila
    model belum tersedia.
    """
    model = st.session_state.get("model")
    feature_cols = st.session_state.get("feature_cols")
    if model is None or not feature_cols:
        return None

    label_encoders = st.session_state.get("label_encoders") or {}
    target_encoder = st.session_state.get("target_encoder")

    x_vector = []
    for col in feature_cols:
        val = new_row.get(col, 0)
        if col in label_encoders:
            le = label_encoders[col]
            val_str = str(val)
            val = int(le.transform([val_str])[0]) if val_str in le.classes_ else 0
        x_vector.append(float(val))

    x_arr = np.array(x_vector, dtype=float).reshape(1, -1)
    kode_pred = model.predict(x_arr)[0]
    proba = model.predict_proba(x_arr)[0]
    label_pred = label_terbaca(kode_pred, target_encoder)
    label_kelas = label_terbaca(model.classes_, target_encoder)

    return {
        "label": label_pred,
        "proba": {kelas: float(p) for kelas, p in zip(label_kelas, proba)},
    }


def tambahkan_kuesioner_ke_dataset(entries: list):
    """Menggabungkan jawaban kuesioner yang tersimpan ke dataset utama (df_raw)."""
    df_baru = pd.DataFrame(entries)
    df_baru = df_baru.drop(columns=["Waktu Pengisian"], errors="ignore")
    if "Tingkat_Kecanduan (Prediksi Sistem)" in df_baru.columns:
        df_baru = df_baru.rename(columns={"Tingkat_Kecanduan (Prediksi Sistem)": "Tingkat_Kecanduan"})

    df_raw = st.session_state.get("df_raw")
    if df_raw is None or df_raw.empty:
        df_baru.insert(0, "No", range(1, len(df_baru) + 1))
        for col in KOLOM_STANDAR_DATASET:
            if col not in df_baru.columns:
                df_baru[col] = pd.NA
        st.session_state["df_raw"] = df_baru[KOLOM_STANDAR_DATASET]
    else:
        nomor_awal = int(pd.to_numeric(df_raw.get("No"), errors="coerce").max() + 1) if "No" in df_raw.columns else len(df_raw) + 1
        df_baru.insert(0, "No", range(nomor_awal, nomor_awal + len(df_baru)))
        gabungan = pd.concat([df_raw, df_baru], ignore_index=True, sort=False)
        st.session_state["df_raw"] = gabungan

    reset_dari_upload_ringan()


def reset_dari_upload_ringan():
    """Reset tahap turunan tanpa menghapus model yang sudah dilatih (khusus penambahan data kuesioner)."""
    for key in ["df_clean", "df_clean_asli"]:
        st.session_state[key] = None


# ===============================================================================
# HALAMAN 0 — DASHBOARD
# ===============================================================================

def halaman_dashboard():
    core.render_page_header(
        "Dashboard",
        "Ringkasan kondisi dataset, model, dan hasil analisis pada Sistem Analisis "
        "Kecanduan Gadget pada Siswa (metode Naive Bayes).",
    )

    df_raw = st.session_state.get("df_raw")
    X_train = st.session_state.get("X_train")
    X_test = st.session_state.get("X_test")
    model = st.session_state.get("model")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        core.render_metric_card(len(df_raw) if df_raw is not None else "-", "Total Data Siswa")
    with c2:
        core.render_metric_card(len(X_train) if X_train is not None else "-", "Data Training")
    with c3:
        core.render_metric_card(len(X_test) if X_test is not None else "-", "Data Testing")
    with c4:
        core.render_metric_card("Tersedia" if model is not None else "Belum Dibuat",
                                 "Status Model", small_text=True)

    tahapan = [
        ("Dataset", df_raw is not None),
        ("Preprocessing", st.session_state.get("df_clean") is not None),
        ("Training", model is not None),
        ("Klasifikasi", st.session_state.get("y_pred") is not None),
        ("Evaluasi", st.session_state.get("metrics") is not None),
    ]
    jumlah_selesai = sum(1 for _, sel in tahapan if sel)
    st.markdown(
        f"<div style='margin:20px 0 4px 0;font-size:.82rem;color:#5B6472;font-weight:600;'>"
        f"Progres alur kerja sistem: {jumlah_selesai} dari {len(tahapan)} tahap selesai</div>",
        unsafe_allow_html=True,
    )
    st.progress(jumlah_selesai / len(tahapan))
    st.write("")

    col_chart, col_status = st.columns([3, 2])

    with col_chart:
        with st.container(border=True):
            core.render_section_title("Distribusi Tingkat Kecanduan Gadget", rapat=True)

            label_series = None
            if st.session_state.get("df_clean_asli") is not None and st.session_state.get("target_col"):
                label_series = st.session_state["df_clean_asli"][st.session_state["target_col"]]
            elif df_raw is not None:
                kolom_mirip = [c for c in df_raw.columns if "kecandu" in str(c).lower()]
                if kolom_mirip:
                    label_series = df_raw[kolom_mirip[0]]

            if label_series is not None and len(label_series) > 0:
                urutan_prioritas = ["Rendah", "Sedang", "Tinggi"]
                counts = label_series.astype(str).value_counts()
                kategori = [k for k in urutan_prioritas if k in counts.index] + \
                           [k for k in counts.index if k not in urutan_prioritas]
                nilai = [int(counts[k]) for k in kategori]
                warna = [core.WARNA_KATEGORI.get(k, core.WARNA_UTAMA) for k in kategori]

                fig, ax = plt.subplots(figsize=(6.2, 3.3))
                bars = ax.bar(kategori, nilai, color=warna, width=0.5, zorder=3)
                for b, n in zip(bars, nilai):
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(nilai) * 0.02,
                            str(n), ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1F2937")
                ax.set_ylabel("Jumlah Siswa", fontsize=9, color="#5B6472")
                for spine in ("top", "right", "left"):
                    ax.spines[spine].set_visible(False)
                ax.spines["bottom"].set_color("#D5DBE5")
                ax.tick_params(axis="both", labelsize=9.5, colors="#5B6472")
                ax.yaxis.grid(True, color="#E5E9F0", zorder=0)
                ax.set_axisbelow(True)
                fig.patch.set_alpha(0)
                ax.set_facecolor("none")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Belum ada data yang dapat ditampilkan. Unggah dataset dan selesaikan "
                        "preprocessing terlebih dahulu untuk melihat grafik distribusi.")

    with col_status:
        with st.container(border=True):
            core.render_section_title("Status Proses Sistem", rapat=True)
            first_pending = next((i for i, (_, sel) in enumerate(tahapan) if not sel), None)
            for i, (nama, selesai) in enumerate(tahapan):
                core.render_step(i + 1, nama, selesai, aktif=(i == first_pending))

    col_act, col_nav = st.columns([3, 2])

    with col_act:
        with st.container(border=True):
            core.render_section_title("Aktivitas Terakhir", rapat=True)
            log = st.session_state.get("activity_log") or []
            if log:
                for item in log:
                    core.render_activity_item(
                        item.get("pesan", ""),
                        f"{item.get('tanggal', '')} {item.get('waktu', '')}",
                    )
            else:
                st.info("Belum ada aktivitas. Mulai dengan mengunggah dataset pada menu **Data** (tab Upload Dataset).")

    with col_nav:
        with st.container(border=True):
            core.render_section_title("Alur Kerja Sistem", rapat=True)
            st.markdown(
                "1. **Data**: unggah dataset, tinjau data siswa, dan lakukan preprocessing.\n"
                "2. **Training & Model**: bagi data dan latih model Naive Bayes.\n"
                "3. **Klasifikasi & Evaluasi**: klasifikasikan data dan ukur kinerja model.\n"
                "4. **Laporan / Hasil Akhir**: tinjau rangkuman dan unduh laporan."
            )

    st.divider()
    st.markdown(
        "<p class='footnote'>Catatan: Algoritma Naive Bayes pada sistem ini diimplementasikan "
        "secara manual (from scratch) menggunakan Python dan NumPy, tanpa memakai fungsi "
        "fit/predict siap pakai dari scikit-learn. Library scikit-learn hanya digunakan untuk "
        "keperluan pendukung, yaitu pembagian data latih/uji dan perhitungan metrik evaluasi.</p>",
        unsafe_allow_html=True,
    )


# ===============================================================================
# HALAMAN 1 — UPLOAD DATASET
# ===============================================================================

def format_ukuran(byte_size: int) -> str:
    for satuan in ["B", "KB", "MB", "GB"]:
        if byte_size < 1024:
            return f"{byte_size:.0f} {satuan}" if satuan == "B" else f"{byte_size:.1f} {satuan}"
        byte_size /= 1024
    return f"{byte_size:.1f} TB"


def konten_upload_dataset():
    st.markdown("##### Unggah data hasil kuesioner siswa (.xlsx / .xls)")
    st.caption("Sistem mendeteksi baris header, membersihkan baris judul dan catatan kaki, "
               "serta mendukung file dengan banyak sheet (misalnya per kelas).")

    with st.expander("Format dataset yang didukung", expanded=False):
        st.markdown("""
        - File Excel **.xlsx** atau **.xls**.
        - Boleh berisi **satu atau beberapa sheet** (mis. data per kelas) — sheet berisi
          teks instruksi/petunjuk akan dilewati.
        - Baris header (nama kolom) **tidak harus di baris pertama** — sistem akan
          mencoba mendeteksinya meskipun terdapat judul instrumen di atasnya.
        - Setiap baris merepresentasikan **1 siswa**, berisi kolom identitas (mis.
          `Nama`, `Kelas`), kolom **fitur** (skor kuesioner), dan idealnya 1 kolom
          **label/target** tingkat kecanduan (mis. `Rendah` / `Sedang` / `Tinggi`).
        - Jika dataset **belum punya kolom label**, tapi memiliki kolom skor total,
          sistem dapat membuat kolom kategori tersebut berdasarkan skor total.
        """)

        contoh_path = os.path.join(BASE_DIR, "data", "contoh_dataset.xlsx")
        if os.path.exists(contoh_path):
            with open(contoh_path, "rb") as f:
                st.download_button(
                    "Unduh Contoh Dataset (contoh_dataset.xlsx)",
                    data=f.read(),
                    file_name="contoh_dataset_kecanduan_gadget.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    st.divider()

    st.markdown('<div class="section-title">Pilih atau Seret File Dataset</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Seret dan lepas file Excel di sini, atau klik untuk memilih file",
        type=["xlsx", "xls"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        st.markdown(f"""
        <div class="file-card">
            <div>
                <div class="name">{uploaded_file.name}</div>
                <div class="meta">Ukuran: {format_ukuran(uploaded_file.size)} • Format: {uploaded_file.type or 'Excel'}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    df_kandidat = None

    if uploaded_file is not None:
        try:
            dict_sheet = core.baca_semua_sheet(uploaded_file)
        except Exception as e:
            st.error(f"Gagal membaca file Excel: {e}")
            dict_sheet = {}

        if not dict_sheet:
            st.markdown(
                '<span class="badge badge-bad">Sheet data tidak terdeteksi</span>',
                unsafe_allow_html=True,
            )
            st.warning("Tidak ditemukan sheet berisi data tabel yang valid pada file ini. "
                       "Pastikan minimal ada 1 sheet dengan baris header (nama kolom) dan data siswa.")
        else:
            st.markdown('<div class="section-title">Pemilihan Sheet Data</div>', unsafe_allow_html=True)

            nama_sheet_valid = list(dict_sheet.keys())
            st.markdown(
                f'<span class="badge badge-ok">{len(nama_sheet_valid)} sheet data terdeteksi: '
                f'{", ".join(nama_sheet_valid)}</span>',
                unsafe_allow_html=True,
            )

            if len(dict_sheet) == 1:
                df_kandidat = list(dict_sheet.values())[0]
            elif core.sheet_bisa_digabung(dict_sheet):
                gabung = st.checkbox(
                    f"Gabungkan semua sheet ({', '.join(nama_sheet_valid)}) menjadi satu dataset",
                    value=True,
                    help="Semua sheet memiliki struktur kolom yang sama, sehingga aman digabung.",
                )
                if gabung:
                    df_kandidat = core.gabungkan_sheet(dict_sheet)
                else:
                    pilihan_sheet = st.selectbox("Pilih salah satu sheet", options=nama_sheet_valid)
                    df_kandidat = dict_sheet[pilihan_sheet]
            else:
                st.info("Struktur kolom antar sheet berbeda, silakan pilih salah satu sheet.")
                pilihan_sheet = st.selectbox("Pilih sheet yang akan digunakan", options=nama_sheet_valid)
                df_kandidat = dict_sheet[pilihan_sheet]

    buat_label_aktif = False
    kolom_skor_terpilih = None

    if df_kandidat is not None:
        st.markdown('<div class="section-title">Validasi Struktur Dataset</div>', unsafe_allow_html=True)

        baris_ok = df_kandidat.shape[0] > 0
        kolom_ok = df_kandidat.shape[1] >= 2
        kolom_target = core.cari_kolom_target_kategorikal(df_kandidat)
        kolom_skor = core.cari_kolom_skor_numerik(df_kandidat)

        lencana = []
        lencana.append('<span class="badge badge-ok">Format file valid</span>')
        lencana.append('<span class="badge badge-ok">Data ditemukan</span>' if baris_ok
                        else '<span class="badge badge-bad">Tidak ada baris data</span>')
        lencana.append('<span class="badge badge-ok">Struktur kolom memadai</span>' if kolom_ok
                        else '<span class="badge badge-bad">Kolom kurang dari 2</span>')
        if kolom_target:
            lencana.append(f'<span class="badge badge-ok">Kolom label terdeteksi: {kolom_target}</span>')
        else:
            lencana.append('<span class="badge badge-warn">Kolom label belum terdeteksi</span>')
        st.markdown("".join(lencana), unsafe_allow_html=True)

        if not kolom_target and kolom_skor:
            st.write("")
            buat_label_aktif = st.checkbox(
                f"Buat kolom kategori **Tingkat_Kecanduan** (Rendah/Sedang/Tinggi) "
                f"dari kolom skor **'{kolom_skor}'**",
                value=True,
                help="Nilai kolom skor akan dibagi menjadi 3 kategori dengan rentang yang sama besar.",
            )
            kolom_skor_terpilih = kolom_skor
        elif not kolom_target and not kolom_skor:
            st.info("Belum ditemukan kolom label maupun kolom skor total. Anda tetap dapat "
                    "melanjutkan dan memilih kolom target secara manual di menu "
                    "**Preprocessing Data**.")


        df_final = df_kandidat.copy()
        if buat_label_aktif and kolom_skor_terpilih:
            df_final["Tingkat_Kecanduan"] = core.buat_label_kategori(df_final, kolom_skor_terpilih)

        st.markdown('<div class="section-title">Pratinjau Data</div>', unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            core.render_metric_mini(len(df_final), "Jumlah Baris")
        with m2:
            core.render_metric_mini(len(df_final.columns), "Jumlah Kolom")
        with m3:
            core.render_metric_mini(int(df_final.isnull().sum().sum()), "Missing Value")

        st.write("")
        st.dataframe(df_final.head(20), use_container_width=True)

        st.markdown('<div class="section-title">Simpan Dataset</div>', unsafe_allow_html=True)
        st.caption("Klik tombol di bawah untuk menyimpan dataset ini dan menggunakannya pada "
                   "tahap-tahap analisis selanjutnya. Tahap yang sudah pernah dijalankan sebelumnya "
                   "(preprocessing, training, dst.) akan direset.")

        jalankan_analisis_aktif = st.checkbox(
            "Jalankan seluruh tahapan analisis setelah data diimpor "
            "(Preprocessing → Training → Klasifikasi → Evaluasi)",
            value=True,
            help="Kolom target dan fitur dikenali dari struktur data, lalu seluruh tahap dijalankan "
                 "dengan pengaturan standar. Pengaturan tersebut dapat diubah kapan saja melalui "
                 "menu Preprocessing dan Training.",
        )

        kolom_sesuai_kuesioner = all(c in df_final.columns for c in KOLOM_FITUR_KUESIONER)
        simpan_ke_db_aktif = st.checkbox(
            "Simpan dataset pada penyimpanan lokal",
            value=kolom_sesuai_kuesioner,
            disabled=not kolom_sesuai_kuesioner,
            help="Hanya tersedia jika struktur kolom dataset sesuai instrumen kuesioner (P1-P15). "
                 "Data akan digabungkan dengan jawaban yang tersimpan secara lokal.",
        )
        if not kolom_sesuai_kuesioner:
            st.caption("Struktur kolom dataset ini tidak sama persis dengan instrumen kuesioner "
                       "(P1-P15), sehingga tidak dapat disimpan ke penyimpanan lokal — dataset tetap "
                       "dapat dipakai untuk analisis di sesi ini seperti biasa.")

        if st.button("Upload / Import Dataset", type="primary"):
            try:
                if df_final.empty:
                    st.error("Gagal mengimpor: dataset kosong setelah diproses.")
                else:
                    st.session_state["df_raw"] = df_final
                    reset_dari_upload()

                    st.success(f"Dataset berhasil diimpor! Total {len(df_final)} baris, "
                               f"{len(df_final.columns)} kolom siap digunakan.")
                    core.log_activity(f"Mengunggah dataset baru ({len(df_final)} baris, "
                                       f"{len(df_final.columns)} kolom)")

                    if simpan_ke_db_aktif and kolom_sesuai_kuesioner:
                        with st.spinner("Menyimpan dataset..."):
                            berhasil, gagal = database.simpan_banyak_jawaban(df_final)
                        if berhasil:
                            st.success(f"{berhasil} baris berhasil disimpan permanen ke database"
                                       + (f" ({gagal} gagal)." if gagal else "."))
                            core.log_activity(f"Menyimpan {berhasil} baris dataset ke database")
                        else:
                            st.warning("Tidak ada data yang tersimpan ke database — periksa "
                                       "kembali koneksi Supabase.")

                    if jalankan_analisis_aktif:
                        with st.spinner("Menjalankan analisis (preprocessing → training → evaluasi)..."):
                            try:
                                ringkasan = core.jalankan_pipeline_analisis(df_final)
                            except ValueError as e:
                                st.warning(f"Analisis tidak dapat dilanjutkan: {e}")
                            else:
                                m = ringkasan["metrics"]
                                st.success(
                                    f"Analisis selesai. Kolom target yang digunakan: "
                                    f"**{ringkasan['target_col']}** • {ringkasan['jumlah_train']} data "
                                    f"training, {ringkasan['jumlah_test']} data testing • "
                                    f"Accuracy model: **{m['Accuracy']*100:.1f}%**."
                                )
                                st.info("Lihat hasil lengkapnya di menu **Dashboard**, **Klasifikasi & Evaluasi**, "
                                        "atau **Laporan / Hasil Akhir** melalui sidebar.")
            except Exception as e:
                st.error(f"Gagal mengimpor dataset: {e}")

    if st.session_state.get("df_raw") is not None:
        df_tersimpan = st.session_state["df_raw"]
        st.success(f"Dataset aktif saat ini: **{len(df_tersimpan)} baris, "
                   f"{len(df_tersimpan.columns)} kolom**. Lanjutkan ke tab **Data Siswa** di atas.")
    elif uploaded_file is None:
        st.warning("Belum ada dataset yang diunggah.")


# ===============================================================================
# HALAMAN 2 — DATA SISWA
# ===============================================================================

def konten_isi_kuesioner():
    core.render_section_title("Petunjuk Pengisian")
    st.markdown(
        "Kuesioner ini disusun berdasarkan **Instrumen Penelitian Kecanduan Gadget pada Siswa SMP "
        "Kelas VII, VIII, dan IX** yang terdiri atas 15 pernyataan. Bacalah setiap pernyataan "
        "dengan cermat, kemudian pilih **satu** huruf jawaban yang paling sesuai dengan kondisi Anda. "
        "Tidak ada jawaban benar atau salah, dan seluruh pernyataan wajib dijawab."
    )
    baris_skala = "".join(
        f'<tr><td class="huruf">{h}</td><td>{ket}</td></tr>' for h, ket in KETERANGAN_JAWABAN
    )
    st.markdown(
        '<table class="skala-tabel"><thead><tr>'
        '<th style="width:90px;text-align:center;">Pilihan</th><th>Keterangan</th>'
        f'</tr></thead><tbody>{baris_skala}</tbody></table>',
        unsafe_allow_html=True,
    )

    versi = st.session_state["versi_form"]

    with st.form(f"form_kuesioner_{versi}"):
        core.render_section_title("Data Responden")
        c1, c2 = st.columns(2)
        with c1:
            nama = st.text_input("Nama Lengkap")
            kelas = st.selectbox("Kelas", options=[7, 8, 9], format_func=lambda x: f"Kelas {x}")
        with c2:
            jenis_kelamin = st.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"], horizontal=True)
            usia = st.number_input("Usia (tahun)", min_value=10, max_value=18, value=13, step=1)

        core.render_section_title("Daftar Pernyataan")
        jawaban_huruf = {}
        for i, pernyataan in enumerate(DAFTAR_PERNYATAAN_KUESIONER, start=1):
            with st.container(border=True):
                jawaban_huruf[f"P{i}"] = st.radio(
                    f"**{i}. {pernyataan}**",
                    options=OPSI_JAWABAN, index=None, horizontal=True,
                    format_func=lambda h: f"{h}. {TEKS_JAWABAN[h]}",
                    key=f"kuesioner_p{i}_{versi}",
                )

        submitted = st.form_submit_button("Kirim Jawaban", type="primary", use_container_width=True)

    if submitted:
        belum_dijawab = [kode[1:] for kode, huruf in jawaban_huruf.items() if huruf is None]
        if not nama.strip():
            st.error("Nama lengkap wajib diisi terlebih dahulu.")
        elif belum_dijawab:
            st.error("Masih terdapat pernyataan yang belum dijawab, yaitu nomor: "
                     + ", ".join(belum_dijawab) + ".")
        else:
            jawaban = {kode: SKOR_JAWABAN[huruf] for kode, huruf in jawaban_huruf.items()}
            total_skor = sum(jawaban.values())

            new_row = {
                "Nama": nama.strip(),
                "Kelas": int(kelas),
                "Jenis Kelamin": jenis_kelamin,
                "Usia": int(usia),
            }
            new_row.update(jawaban)
            new_row["Total Skor"] = total_skor

            hasil = hitung_klasifikasi(new_row)
            new_row["Tingkat_Kecanduan (Prediksi Sistem)"] = hasil["label"] if hasil else "Menunggu Model Dilatih"
            new_row["Waktu Pengisian"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

            if st.session_state.get("kuesioner_entries") is None:
                st.session_state["kuesioner_entries"] = []
            st.session_state["kuesioner_entries"].append(new_row)

            tersimpan_permanen = database.simpan_satu_jawaban(new_row)
            core.log_activity(f"Kuesioner diisi oleh {new_row['Nama']}")

            st.session_state["hasil_kuesioner_terakhir"] = {
                "nama": new_row["Nama"],
                "total_skor": total_skor,
                "permanen": tersimpan_permanen,
                "galat": "" if tersimpan_permanen else database.galat_terakhir(),
                "hasil": hasil,
            }
            # Naikkan versi form agar seluruh isian dikosongkan setelah pengiriman berhasil.
            st.session_state["versi_form"] += 1
            st.rerun()

    terakhir = st.session_state.get("hasil_kuesioner_terakhir")
    # Tidak menampilkan skor, hasil klasifikasi, atau perhitungan kepada siswa.
    # Konfirmasi khusus ditampilkan pada halaman terpisah melalui halaman_portal_siswa.



def konten_data_siswa():
    st.markdown("##### Data Siswa")
    st.caption("Menampilkan dan memeriksa data siswa yang telah diunggah/tersimpan.")

    status_db = database.status_koneksi()
    col_status, col_muat = st.columns([3, 1])
    with col_status:
        if status_db["terhubung"]:
            st.markdown('<span class="badge badge-ok">Database tersambung — jawaban kuesioner siswa '
                         'tersimpan permanen di sana</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-warn">Penyimpanan lokal belum tersedia</span>',
                        unsafe_allow_html=True)
            st.caption(status_db["pesan"])
            if st.button("Periksa Koneksi Ulang"):
                database.segarkan_status()
                st.rerun()
    with col_muat:
        if st.button("Perbarui Jawaban Siswa", use_container_width=True, disabled=not status_db["terhubung"]):
            df_db = database.ambil_semua_jawaban()
            if df_db.empty:
                st.warning(database.galat_terakhir() or "Belum ada jawaban kuesioner yang tersimpan di database.")
            else:
                df_db = core.lengkapi_label_dari_skor(df_db)
                st.session_state["df_raw"] = df_db
                reset_dari_upload()
                core.log_activity(f"Memuat {len(df_db)} jawaban kuesioner dari database")
                st.success(f"Berhasil memuat {len(df_db)} data dari database!")
                st.rerun()

    # Saat admin membuka menu, data jawaban siswa langsung diambil dari database.
    # Tombol manual tetap tersedia sebagai pilihan untuk memperbarui data.
    if st.session_state.get("df_raw") is None and status_db["terhubung"]:
        sinkronkan_jawaban_database(silent=True)

    if st.session_state.get("df_raw") is None:
        st.info("Belum ada jawaban siswa. Siswa dapat mengisi melalui menu Kuesioner Siswa.")
        return

    df = st.session_state["df_raw"]

    st.subheader("Pencarian & Filter")
    col1, col2 = st.columns([2, 1])
    with col1:
        kata_kunci = st.text_input("Cari berdasarkan nilai pada kolom apa saja (mis. nama siswa)")
    with col2:
        kolom_filter = st.selectbox("Filter berdasarkan kolom (opsional)", ["(Tidak ada)"] + list(df.columns))

    df_tampil = df.copy()

    if kolom_filter != "(Tidak ada)":
        nilai_unik = ["(Semua)"] + sorted(df[kolom_filter].dropna().astype(str).unique().tolist())
        pilihan = st.selectbox(f"Pilih nilai untuk kolom '{kolom_filter}'", nilai_unik)
        if pilihan != "(Semua)":
            df_tampil = df_tampil[df_tampil[kolom_filter].astype(str) == pilihan]

    if kata_kunci:
        mask = df_tampil.astype(str).apply(lambda row: row.str.contains(kata_kunci, case=False, na=False)).any(axis=1)
        df_tampil = df_tampil[mask]

    st.divider()
    st.subheader(f"Tabel Data Siswa ({len(df_tampil)} dari {len(df)} baris)")
    st.dataframe(df_tampil, use_container_width=True)

    st.divider()
    st.subheader("Statistik Ringkas")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Siswa", len(df))
    c2.metric("Jumlah Kolom", len(df.columns))
    c3.metric("Data Duplikat", int(df.duplicated().sum()))
    c4.metric("Total Missing Value", int(df.isnull().sum().sum()))

    with st.expander("Lihat statistik deskriptif kolom numerik"):
        kolom_numerik = df.select_dtypes(include="number")
        if not kolom_numerik.empty:
            st.dataframe(kolom_numerik.describe().T, use_container_width=True)
        else:
            st.info("Tidak ada kolom numerik pada dataset ini.")

    with st.expander("Lihat tipe data & jumlah nilai kosong per kolom"):
        info_df = df.dtypes.astype(str).reset_index()
        info_df.columns = ["Kolom", "Tipe Data"]
        info_df["Jumlah Missing"] = df.isnull().sum().values
        st.dataframe(info_df, use_container_width=True)


# ===============================================================================
# HALAMAN 3 — PREPROCESSING DATA
# ===============================================================================

def konten_preprocessing():
    st.markdown("##### Preprocessing Data")
    st.caption("Membersihkan, memeriksa, dan mempersiapkan data sebelum dianalisis.")

    if st.session_state.get("df_raw") is None:
        st.warning("Belum ada dataset. Silakan unggah dataset terlebih dahulu di tab **Upload Dataset** di atas.")
        return

    df_raw = st.session_state["df_raw"]

    st.subheader("1. Pemeriksaan Missing Value")
    st.dataframe(core.ringkasan_missing_value(df_raw), use_container_width=True)

    metode_missing = st.radio(
        "Metode penanganan missing value",
        options=["hapus_baris", "isi_rata_median"],
        format_func=lambda x: {
            "hapus_baris": "Hapus baris yang memiliki nilai kosong",
            "isi_rata_median": "Isi dengan median (angka) dan modus (kategori)",
        }[x],
        horizontal=True,
    )

    hapus_dup = st.checkbox("Hapus data duplikat", value=True)

    st.divider()

    st.subheader("2. Pemilihan Kolom Target dan Fitur")

    kolom_identitas_terdeteksi = core.deteksi_kolom_identitas(df_raw)
    kolom_skor_terdeteksi = [c for c in core.kolom_skor_atau_rekap(df_raw)]
    semua_kolom = list(df_raw.columns)

    target_col = st.selectbox(
        "Pilih kolom **target/label** (tingkat kecanduan gadget)",
        options=semua_kolom,
        index=(len(semua_kolom) - 1) if semua_kolom else 0,
        help="Kolom ini berisi kategori yang ingin diprediksi, misalnya Rendah/Sedang/Tinggi.",
    )

    kolom_kandidat_fitur = [c for c in semua_kolom if c != target_col]
    kolom_dikecualikan = set(kolom_identitas_terdeteksi) | (set(kolom_skor_terdeteksi) - {target_col})
    default_fitur = [c for c in kolom_kandidat_fitur if c not in kolom_dikecualikan]

    if kolom_skor_terdeteksi and any(c != target_col for c in kolom_skor_terdeteksi):
        st.caption(
            f"Kolom skor/rekap ({', '.join(c for c in kolom_skor_terdeteksi if c != target_col)}) "
            f"tidak dipilih sebagai fitur secara bawaan karena kemungkinan menjadi sumber label "
            f"'{target_col}' (mencegah data leakage). Pilih secara manual bila memang ingin diikutkan."
        )

    feature_cols = st.multiselect(
        "Pilih kolom **fitur** yang digunakan untuk analisis "
        "(kolom identitas seperti Nama/Kelas tidak dipilih secara bawaan; ubah bila diperlukan)",
        options=kolom_kandidat_fitur,
        default=default_fitur,
    )

    if not feature_cols:
        st.error("Pilih minimal 1 kolom fitur untuk melanjutkan.")
        return

    st.divider()

    if st.button("Jalankan Preprocessing", type="primary"):
        df_bersih = core.tangani_missing_value(df_raw, metode=metode_missing)
        if hapus_dup:
            df_bersih = core.hapus_duplikat(df_bersih)

        if df_bersih.empty:
            st.error("Setelah preprocessing, data menjadi kosong. Coba gunakan metode "
                     "'Isi dengan median dan modus' agar data tidak terlalu banyak terbuang.")
            return

        df_encoded, label_encoders, target_encoder = core.encode_fitur_dan_target(
            df_bersih, feature_cols, target_col
        )

        st.session_state["df_clean"] = df_encoded
        st.session_state["feature_cols"] = feature_cols
        st.session_state["target_col"] = target_col
        st.session_state["label_encoders"] = label_encoders
        st.session_state["target_encoder"] = target_encoder
        st.session_state["df_clean_asli"] = df_bersih

        reset_dari_preprocessing()

        st.success(f"Preprocessing selesai! Data akhir: {len(df_encoded)} baris "
                   f"(dari {len(df_raw)} baris awal).")
        core.log_activity(f"Menyelesaikan preprocessing data ({len(df_encoded)} baris siap dianalisis)")

    if st.session_state.get("df_clean") is not None:
        st.divider()
        st.subheader("Hasil Preprocessing (Data Siap Dianalisis)")
        st.dataframe(st.session_state["df_clean"].head(20), use_container_width=True)

        if st.session_state["label_encoders"]:
            with st.expander("Lihat pemetaan encoding kolom kategorik → angka"):
                for col, le in st.session_state["label_encoders"].items():
                    mapping = {kelas: int(kode) for kode, kelas in enumerate(le.classes_)}
                    st.write(f"**{col}**:", mapping)

        if st.session_state["target_encoder"] is not None:
            le = st.session_state["target_encoder"]
            mapping = {kelas: int(kode) for kode, kelas in enumerate(le.classes_)}
            st.write("**Pemetaan Kelas Target:**", mapping)

        st.info("Data siap. Lanjutkan ke menu **Training & Model** (tab Data Training & Testing).")


# ===============================================================================
# HALAMAN 4 — DATA TRAINING & TESTING
# ===============================================================================

def konten_training_testing():
    st.markdown("##### Data Training & Testing")
    st.caption("Membagi dataset menjadi data pelatihan (training) dan data pengujian (testing).")

    if st.session_state.get("df_clean") is None:
        st.warning("Data belum diproses. Silakan selesaikan tab **Preprocessing Data** (menu Data) terlebih dahulu.")
        return

    df_clean = st.session_state["df_clean"]
    feature_cols = st.session_state["feature_cols"]
    target_col = st.session_state["target_col"]

    st.subheader("Pengaturan Pembagian Data")
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Proporsi Data Testing", min_value=0.1, max_value=0.5, value=0.2, step=0.05,
                               help="Contoh: 0.2 artinya 20% data untuk testing, 80% untuk training.")
    with col2:
        random_state = st.number_input("Random State (agar hasil pembagian konsisten)", min_value=0, value=42, step=1)

    stratify_option = st.checkbox(
        "Gunakan Stratified Split (menjaga proporsi tiap kategori sama di training & testing)",
        value=True,
    )

    if st.button("Bagi Data", type="primary"):
        X = df_clean[feature_cols].values
        y = df_clean[target_col].values
        idx_all = df_clean.index.values

        stratify_arr = y if stratify_option else None
        try:
            X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
                X, y, idx_all, test_size=test_size, random_state=int(random_state), stratify=stratify_arr
            )
        except ValueError as e:
            st.warning(f"Stratified split gagal ({e}), melakukan split biasa tanpa stratifikasi.")
            X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
                X, y, idx_all, test_size=test_size, random_state=int(random_state)
            )

        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test
        st.session_state["train_idx"] = train_idx
        st.session_state["test_idx"] = test_idx

        reset_dari_split()

        st.success(f"Data berhasil dibagi: {len(X_train)} data training, {len(X_test)} data testing.")
        core.log_activity(f"Membagi data menjadi {len(X_train)} data training & {len(X_test)} data testing")

    if st.session_state.get("X_train") is not None:
        st.divider()
        st.subheader("Ringkasan Hasil Pembagian")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Data", len(df_clean))
        c2.metric("Data Training", len(st.session_state["X_train"]))
        c3.metric("Data Testing", len(st.session_state["X_test"]))

        target_encoder = st.session_state.get("target_encoder")

        def _label(arr):
            return label_terbaca(arr, target_encoder)

        st.markdown("**Distribusi Kelas pada Data Training:**")
        train_dist = pd.Series(_label(st.session_state["y_train"])).value_counts()
        st.bar_chart(train_dist)

        st.markdown("**Distribusi Kelas pada Data Testing:**")
        test_dist = pd.Series(_label(st.session_state["y_test"])).value_counts()
        st.bar_chart(test_dist)

        st.info("Data siap dilatih. Lanjutkan ke tab **Perhitungan Naive Bayes** di atas.")


# ===============================================================================
# HALAMAN 5 — PERHITUNGAN NAIVE BAYES
# ===============================================================================

def konten_perhitungan_naive_bayes():
    st.markdown("##### Perhitungan Naive Bayes")
    st.caption("Menghitung probabilitas (prior, mean, varians) dan membangun model klasifikasi Naive Bayes.")

    if st.session_state.get("X_train") is None:
        st.warning("Data belum dibagi. Silakan selesaikan tab **Data Training & Testing** di atas terlebih dahulu.")
        return

    st.markdown(r"""
Algoritma **Gaussian Naive Bayes** memprediksi kelas berdasarkan teorema Bayes dengan
asumsi setiap fitur berdistribusi normal (Gaussian) pada tiap kelas:

$$P(C_k \mid x) \; \propto \; P(C_k) \prod_{i=1}^{n} P(x_i \mid C_k)$$

$$P(x_i \mid C_k) = \dfrac{1}{\sqrt{2\pi\sigma_{k}^2}} \exp\!\left(-\dfrac{(x_i-\mu_{k})^2}{2\sigma_{k}^2}\right)$$

Tahap pelatihan (*training*) menghitung 3 komponen dari data training:
- **Prior** P(Ck): proporsi jumlah data tiap kelas.
- **Mean (μ)**: rata-rata tiap fitur pada tiap kelas.
- **Varians (σ²)**: sebaran/variansi tiap fitur pada tiap kelas.
""")

    st.divider()

    if st.button("Latih Model Naive Bayes", type="primary"):
        model = core.GaussianNaiveBayesManual()
        model.fit(
            st.session_state["X_train"],
            st.session_state["y_train"],
            feature_names=st.session_state["feature_cols"],
        )
        st.session_state["model"] = model
        reset_dari_training()
        st.success("Model Naive Bayes berhasil dilatih menggunakan data training!")
        core.log_activity("Melatih model Naive Bayes menggunakan data training")

    model = st.session_state.get("model")
    target_encoder = st.session_state.get("target_encoder")

    if model is not None:
        def _label(kode):
            return label_terbaca(kode, target_encoder)

        st.divider()
        st.subheader("1. Probabilitas Prior Tiap Kelas — P(Kelas)")
        prior_df = model.get_prior_table()
        prior_df_tampil = prior_df.copy()
        prior_df_tampil["Kelas"] = prior_df_tampil["Kelas"].apply(_label)
        st.dataframe(prior_df_tampil, use_container_width=True)

        st.divider()
        st.subheader("2. Mean (μ) dan Varians (σ²) Tiap Fitur per Kelas")
        mv_df = model.get_mean_var_table()
        mv_df_tampil = mv_df.copy()
        mv_df_tampil["Kelas"] = mv_df_tampil["Kelas"].apply(_label)
        st.dataframe(mv_df_tampil, use_container_width=True)

        st.divider()
        st.subheader("3. Contoh Perhitungan Manual (1 Data Uji)")
        st.caption("Menunjukkan rincian perhitungan log-probabilitas untuk salah satu data testing sebagai ilustrasi.")

        if st.session_state.get("X_test") is not None and len(st.session_state["X_test"]) > 0:
            idx_contoh = st.number_input(
                "Pilih indeks data testing untuk ditampilkan rinciannya",
                min_value=0, max_value=len(st.session_state["X_test"]) - 1, value=0, step=1,
            )
            x_contoh = st.session_state["X_test"][int(idx_contoh)]

            with st.expander("Lihat nilai fitur data yang dipilih"):
                for fname, val in zip(st.session_state["feature_cols"], x_contoh):
                    st.write(f"- **{fname}** = {val}")

            detail_df, pred_kelas = model.explain_single_prediction(x_contoh)
            detail_tampil = detail_df.copy()
            detail_tampil["Kelas"] = detail_tampil["Kelas"].apply(_label)
            st.dataframe(detail_tampil, use_container_width=True)

            st.success(f"Prediksi kelas untuk data ini: **{_label(pred_kelas)}** "
                       f"(kelas dengan posterior tertinggi)")

        st.info("Model sudah terlatih. Lanjutkan ke menu **Klasifikasi & Evaluasi** (tab Proses Klasifikasi).")


# ===============================================================================
# HALAMAN 6 — PROSES KLASIFIKASI
# ===============================================================================

def konten_proses_klasifikasi():
    st.markdown("##### Proses Klasifikasi")
    st.caption("Menentukan tingkat kecanduan gadget siswa berdasarkan hasil perhitungan model.")

    if st.session_state.get("model") is None:
        st.warning("Model belum dilatih. Silakan selesaikan tab **Perhitungan Naive Bayes** (menu Training & Model) terlebih dahulu.")
        return

    model = st.session_state["model"]
    target_encoder = st.session_state.get("target_encoder")

    if st.button("Klasifikasikan Data Testing", type="primary"):
        X_test = st.session_state["X_test"]
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        st.session_state["y_pred"] = y_pred
        st.session_state["y_proba"] = y_proba
        st.session_state["metrics"] = None
        st.session_state["hasil_klasifikasi_df"] = None

        st.success(f"Klasifikasi selesai untuk {len(X_test)} data testing.")
        core.log_activity(f"Menjalankan klasifikasi untuk {len(X_test)} data testing")

    if st.session_state.get("y_pred") is not None:
        st.divider()
        st.subheader("Hasil Prediksi Data Testing")

        y_pred = st.session_state["y_pred"]
        y_true = st.session_state["y_test"]
        y_proba = st.session_state["y_proba"]

        def _label(arr):
            return label_terbaca(arr, target_encoder)

        tabel = pd.DataFrame({
            "Label Aktual": _label(y_true),
            "Label Prediksi": _label(y_pred),
        })
        tabel["Status"] = ["Benar" if a == p else "Salah"
                            for a, p in zip(tabel["Label Aktual"], tabel["Label Prediksi"])]

        kelas_urut = _label(model.classes_)
        for i, kelas in enumerate(kelas_urut):
            tabel[f"P({kelas})"] = y_proba[:, i].round(4)

        st.dataframe(tabel, use_container_width=True)

        benar = int((tabel["Status"] == "Benar").sum())
        st.metric("Jumlah Prediksi Benar (sementara)", f"{benar} dari {len(tabel)}")

        st.info("Klasifikasi selesai. Lanjutkan ke tab **Evaluasi Model** di atas untuk melihat metrik "
                "kinerja lengkap (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).")


# ===============================================================================
# HALAMAN 7 — EVALUASI MODEL
# ===============================================================================

def konten_evaluasi_model():
    st.markdown("##### Evaluasi Model")
    st.caption("Mengukur kinerja model menggunakan Accuracy, Precision, Recall, F1-Score, dan Confusion Matrix.")

    if st.session_state.get("y_pred") is None:
        st.warning("Belum ada hasil klasifikasi. Silakan selesaikan tab **Proses Klasifikasi** di atas terlebih dahulu.")
        return

    y_true = st.session_state["y_test"]
    y_pred = st.session_state["y_pred"]
    model = st.session_state["model"]
    target_encoder = st.session_state.get("target_encoder")

    label_kelas = (target_encoder.inverse_transform(model.classes_)
                   if target_encoder is not None else model.classes_)

    if st.button("Hitung Evaluasi Model", type="primary"):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=model.classes_)

        metrics = {
            "Accuracy": accuracy,
            "Precision (macro avg)": precision,
            "Recall (macro avg)": recall,
            "F1-Score (macro avg)": f1,
        }
        st.session_state["metrics"] = metrics
        st.session_state["confusion_matrix"] = cm
        st.success("Evaluasi model berhasil dihitung.")
        core.log_activity(f"Menghitung evaluasi model (Accuracy {accuracy*100:.1f}%)")

    metrics = st.session_state.get("metrics")

    if metrics is not None:
        st.divider()
        st.subheader("Ringkasan Metrik Evaluasi")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics['Accuracy']*100:.2f}%")
        c2.metric("Precision", f"{metrics['Precision (macro avg)']*100:.2f}%")
        c3.metric("Recall", f"{metrics['Recall (macro avg)']*100:.2f}%")
        c4.metric("F1-Score", f"{metrics['F1-Score (macro avg)']*100:.2f}%")

        st.divider()
        st.subheader("Confusion Matrix")
        cm = st.session_state["confusion_matrix"]

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap=core.CMAP_FORMAL)
        ax.set_xticks(range(len(label_kelas)))
        ax.set_yticks(range(len(label_kelas)))
        ax.set_xticklabels(label_kelas, rotation=45, ha="right", color="#1F2937")
        ax.set_yticklabels(label_kelas, color="#1F2937")
        ax.set_xlabel("Label Prediksi", color="#1F2937")
        ax.set_ylabel("Label Aktual", color="#1F2937")
        ax.set_title("Confusion Matrix", color="#1F2937", fontweight="bold")
        fig.patch.set_alpha(0)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                warna_teks = "white" if cm[i, j] > cm.max() / 2 else "#1F2937"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=warna_teks, fontweight="bold")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        _, kolom_cm, _ = st.columns([1, 2, 1])
        with kolom_cm:
            st.pyplot(fig)

        st.divider()
        st.subheader("Laporan Klasifikasi Lengkap (per Kelas)")
        report_dict = classification_report(
            y_true, y_pred, labels=model.classes_, target_names=[str(x) for x in label_kelas],
            output_dict=True, zero_division=0,
        )
        report_df = pd.DataFrame(report_dict).T.round(4)
        st.dataframe(report_df, use_container_width=True)

        st.info("Evaluasi selesai. Lanjutkan ke tab **Hasil Klasifikasi** di atas untuk melihat hasil per siswa.")


# ===============================================================================
# HALAMAN 8 — HASIL KLASIFIKASI
# ===============================================================================

def konten_hasil_klasifikasi():
    st.markdown("##### Hasil Klasifikasi")
    st.caption("Menampilkan hasil klasifikasi tingkat kecanduan gadget setiap siswa.")

    if st.session_state.get("y_pred") is None:
        st.warning("Belum ada hasil klasifikasi. Silakan selesaikan tab **Proses Klasifikasi** di atas terlebih dahulu.")
        return

    target_encoder = st.session_state.get("target_encoder")
    model = st.session_state["model"]

    def _label(arr):
        return label_terbaca(arr, target_encoder)

    tab1, tab2 = st.tabs(["Hasil pada Data Testing", "Klasifikasikan Seluruh Data Siswa"])

    with tab1:
        df_asli = st.session_state.get("df_clean_asli")
        test_idx = st.session_state.get("test_idx")

        hasil = pd.DataFrame({
            "Label Aktual": _label(st.session_state["y_test"]),
            "Label Prediksi": _label(st.session_state["y_pred"]),
        })
        hasil["Status"] = ["Sesuai" if a == p else "Tidak Sesuai"
                            for a, p in zip(hasil["Label Aktual"], hasil["Label Prediksi"])]

        if df_asli is not None and test_idx is not None:
            identitas_cols = [c for c in df_asli.columns
                               if c not in st.session_state["feature_cols"] + [st.session_state["target_col"]]]
            if identitas_cols:
                identitas_df = df_asli.loc[test_idx, identitas_cols].reset_index(drop=True)
                hasil = pd.concat([identitas_df, hasil], axis=1)

        st.session_state["hasil_klasifikasi_df"] = hasil

        st.dataframe(hasil, use_container_width=True)

        st.divider()
        st.subheader("Ringkasan Jumlah Siswa per Kategori (Data Testing)")
        ringkasan = hasil["Label Prediksi"].value_counts().reset_index()
        ringkasan.columns = ["Kategori Kecanduan", "Jumlah Siswa"]
        st.session_state["ringkasan_kelas"] = ringkasan

        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(ringkasan, use_container_width=True)
        with c2:
            st.bar_chart(ringkasan.set_index("Kategori Kecanduan"))

        csv_bytes = hasil.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Unduh Hasil (CSV)", data=csv_bytes,
                            file_name="hasil_klasifikasi_testing.csv", mime="text/csv")

    with tab2:
        st.markdown("Menerapkan model yang sudah dilatih ke **seluruh data siswa** "
                    "(bukan hanya data testing), agar setiap siswa memiliki hasil prediksi.")

        if st.button("Klasifikasikan Seluruh Data Siswa"):
            df_clean = st.session_state["df_clean"]
            df_asli = st.session_state["df_clean_asli"]
            feature_cols = st.session_state["feature_cols"]

            X_all = df_clean[feature_cols].values
            pred_all = model.predict(X_all)

            identitas_cols = [c for c in df_asli.columns
                               if c not in feature_cols + [st.session_state["target_col"]]]

            hasil_all = df_asli[identitas_cols].reset_index(drop=True).copy() if identitas_cols else pd.DataFrame()
            hasil_all["Label Aktual"] = _label(df_clean[st.session_state["target_col"]].values)
            hasil_all["Label Prediksi (Model)"] = _label(pred_all)

            st.session_state["hasil_seluruh_data"] = hasil_all
            st.success(f"Berhasil mengklasifikasikan {len(hasil_all)} siswa.")
            core.log_activity(f"Mengklasifikasikan seluruh data ({len(hasil_all)} siswa)")

        if st.session_state.get("hasil_seluruh_data") is not None:
            hasil_all = st.session_state["hasil_seluruh_data"]
            st.dataframe(hasil_all, use_container_width=True)

            ringkasan_all = hasil_all["Label Prediksi (Model)"].value_counts().reset_index()
            ringkasan_all.columns = ["Kategori Kecanduan", "Jumlah Siswa"]
            st.bar_chart(ringkasan_all.set_index("Kategori Kecanduan"))

            csv_bytes_all = hasil_all.to_csv(index=False).encode("utf-8-sig")
            st.download_button("Unduh Hasil Seluruh Data (CSV)", data=csv_bytes_all,
                                file_name="hasil_klasifikasi_seluruh_siswa.csv", mime="text/csv")

    st.info("Lanjutkan ke menu **Laporan / Hasil Akhir** untuk rangkuman dan ekspor laporan.")


# ===============================================================================
# HALAMAN 9 — LAPORAN / HASIL AKHIR
# ===============================================================================

def halaman_laporan_akhir():
    core.render_page_header("Laporan / Hasil Akhir", "Rangkuman hasil analisis dan ekspor laporan")

    if st.session_state.get("metrics") is None or st.session_state.get("hasil_klasifikasi_df") is None:
        st.warning("Lengkapi dahulu tahap **Evaluasi Model** dan **Hasil Klasifikasi** sebelum membuat laporan.")
        return

    metrics = st.session_state["metrics"]
    hasil_df = st.session_state["hasil_klasifikasi_df"]
    ringkasan_kelas = st.session_state.get("ringkasan_kelas")
    if ringkasan_kelas is None:
        ringkasan_kelas = hasil_df["Label Prediksi"].value_counts().reset_index()
        ringkasan_kelas.columns = ["Kategori Kecanduan", "Jumlah Siswa"]

    jumlah_data = len(st.session_state["df_clean"])
    jumlah_train = len(st.session_state["X_train"])
    jumlah_test = len(st.session_state["X_test"])

    st.subheader("1. Ringkasan Dataset")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Data Siswa", jumlah_data)
    c2.metric("Data Training", jumlah_train)
    c3.metric("Data Testing", jumlah_test)

    st.divider()
    st.subheader("2. Ringkasan Evaluasi Model Naive Bayes")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['Accuracy']*100:.2f}%")
    c2.metric("Precision", f"{metrics['Precision (macro avg)']*100:.2f}%")
    c3.metric("Recall", f"{metrics['Recall (macro avg)']*100:.2f}%")
    c4.metric("F1-Score", f"{metrics['F1-Score (macro avg)']*100:.2f}%")

    st.divider()
    st.subheader("3. Rangkuman Jumlah Siswa per Kategori Kecanduan Gadget")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(ringkasan_kelas, use_container_width=True)
    with c2:
        st.bar_chart(ringkasan_kelas.set_index(ringkasan_kelas.columns[0]))

    st.divider()
    st.subheader("4. Tabel Hasil Klasifikasi Lengkap")
    st.dataframe(hasil_df, use_container_width=True)

    st.divider()
    st.subheader("Ekspor Laporan")

    col1, col2 = st.columns(2)

    with col1:
        excel_bytes = core.generate_excel_report(hasil_df, metrics, ringkasan_kelas)
        st.download_button(
            "Unduh Laporan Excel (.xlsx)",
            data=excel_bytes,
            file_name="laporan_analisis_kecanduan_gadget.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    with col2:
        try:
            pdf_bytes = core.generate_pdf_report(
                judul="Laporan Analisis Kecanduan Gadget pada Siswa",
                metrics=metrics,
                ringkasan_kelas=ringkasan_kelas,
                jumlah_data=jumlah_data,
                jumlah_train=jumlah_train,
                jumlah_test=jumlah_test,
            )
            st.download_button(
                "Unduh Laporan PDF (.pdf)",
                data=pdf_bytes,
                file_name="laporan_analisis_kecanduan_gadget.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except ImportError:
            st.warning("Fitur ekspor PDF memerlukan library **fpdf2**. "
                       "Jalankan `pip install fpdf2` lalu muat ulang halaman ini untuk mengaktifkannya. "
                       "Sementara itu, gunakan tombol ekspor Excel di sebelah kiri.")

    st.success("Analisis selesai! Terima kasih telah menggunakan sistem ini.")


# ===============================================================================
# SINKRONISASI DATA DATABASE
# ===============================================================================

def sinkronkan_jawaban_database(silent=False):
    """Mengambil jawaban siswa terbaru dari database ke tampilan admin."""
    status = database.status_koneksi()
    if not status["terhubung"]:
        return False
    df_db = database.ambil_semua_jawaban()
    if df_db.empty:
        return False
    df_db = core.lengkapi_label_dari_skor(df_db)
    st.session_state["df_raw"] = df_db
    reset_dari_upload()
    if not silent:
        st.success(f"{len(df_db)} jawaban siswa berhasil dimuat dari database.")
    return True


def halaman_data():
    core.render_page_header(
        "Data",
        "Mengelola data siswa: unggah dataset Excel (hasil kuesioner), lihat dan telusuri data, "
        "lalu bersihkan (preprocessing) sebelum dianalisis. Jawaban dari sistem Kuesioner Siswa "
        "juga akan muncul di sini.",
    )
    tab1, tab2, tab3 = st.tabs(["Upload Dataset", "Jawaban Siswa", "Preprocessing Data"])
    with tab1:
        konten_upload_dataset()
    with tab2:
        konten_data_siswa()
    with tab3:
        konten_preprocessing()


def halaman_training_model():
    core.render_page_header(
        "Training & Model",
        "Membagi data menjadi training/testing, lalu melihat & melatih model Gaussian Naive Bayes.",
    )
    tab1, tab2 = st.tabs(["Data Training & Testing", "Perhitungan Naive Bayes"])
    with tab1:
        konten_training_testing()
    with tab2:
        konten_perhitungan_naive_bayes()


def halaman_klasifikasi_evaluasi():
    core.render_page_header(
        "Klasifikasi & Evaluasi",
        "Menjalankan klasifikasi pada data testing, mengukur kinerja model, dan melihat "
        "hasil prediksi tiap siswa.",
    )
    tab1, tab2, tab3 = st.tabs(["Proses Klasifikasi", "Evaluasi Model", "Hasil Klasifikasi"])
    with tab1:
        konten_proses_klasifikasi()
    with tab2:
        konten_evaluasi_model()
    with tab3:
        konten_hasil_klasifikasi()


# ===============================================================================
# HALAMAN SISWA — hanya berisi form kuesioner, tanpa akses ke menu admin
# ===============================================================================

def halaman_portal_siswa():
    terakhir = st.session_state.get("hasil_kuesioner_terakhir")
    if terakhir:
        # Halaman konfirmasi estetik setelah pengiriman, tanpa hasil atau perhitungan.
        st.markdown("""
        <div style="max-width:720px;margin:4vh auto 0;background:#ffffff;border:1px solid #dce6f2;
                    border-radius:24px;overflow:hidden;box-shadow:0 18px 45px rgba(28,58,94,.10);">
          <div style="height:9px;background:linear-gradient(90deg,#294d78,#5d85b2,#9cb7d4);"></div>
          <div style="padding:44px 36px 38px;text-align:center;">
            <div style="width:76px;height:76px;margin:0 auto 22px;border-radius:50%;background:#eaf5ef;
                        color:#28734d;display:flex;align-items:center;justify-content:center;font-size:38px;font-weight:700;">✓</div>
            <div style="font-size:12px;letter-spacing:2px;text-transform:uppercase;color:#6e829b;font-weight:700;">Konfirmasi Pengisian</div>
            <h1 style="font-size:30px;color:#526579;margin:12px 0 10px;font-weight:750;">Jawaban Tersimpan</h1>
            <p style="font-size:16px;line-height:1.8;color:#66778b;margin:0 auto;max-width:490px;">
              Terima kasih telah mengisi kuesioner. Jawaban Anda telah diterima dan disimpan.
            </p>
            <div style="margin:26px auto 0;max-width:440px;background:#f4f7fb;border:1px solid #e1e8f0;
                        border-radius:13px;padding:15px 18px;text-align:left;">
              <div style="font-size:12px;color:#74859a;margin-bottom:5px;">Nama responden</div>
              <div style="font-size:17px;color:#526579;font-weight:700;">RESPONDENT_NAME</div>
            </div>
            <p style="font-size:12px;color:#8290a2;margin-top:22px;">Halaman ini hanya sebagai pemberitahuan penyimpanan jawaban.</p>
          </div>
        </div>
        """.replace("RESPONDENT_NAME", __import__('html').escape(str(terakhir.get('nama','Siswa')))), unsafe_allow_html=True)
        if not terakhir.get("permanen"):
            st.warning("Jawaban belum berhasil disimpan ke file lokal. Silakan hubungi guru sebelum meninggalkan halaman.")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Selesai — Kembali ke Halaman Awal", use_container_width=True, type="primary"):
                st.session_state["hasil_kuesioner_terakhir"] = None
                st.session_state["peran"] = None
                st.rerun()
        return

    col_judul, col_keluar = st.columns([4, 1])
    with col_judul:
        core.render_page_header(
            "Kuesioner Siswa",
            "Mohon isi seluruh pernyataan berikut dengan jujur sesuai kondisi Anda sehari-hari.",
        )
    with col_keluar:
        st.write("")
        if st.button("Kembali ke Halaman Awal", use_container_width=True):
            st.session_state["peran"] = None
            st.rerun()

    konten_isi_kuesioner()


# ===============================================================================
# GERBANG LOGIN ADMIN — kata sandi sederhana (lihat ADMIN_PASSWORD di atas)
# ===============================================================================

def halaman_login_admin():
    _, tengah, _ = st.columns([1, 2, 1])
    with tengah:
        core.render_page_header(
            "Masuk sebagai Guru / Administrator",
            "Masukkan kata sandi untuk mengakses menu pengelolaan data dan analisis.",
        )
        with st.container(border=True):
            with st.form("form_login_admin"):
                password = st.text_input("Kata Sandi", type="password")
                masuk = st.form_submit_button("Masuk", type="primary", use_container_width=True)

            if masuk:
                if password == ADMIN_PASSWORD:
                    st.session_state["admin_terautentikasi"] = True
                    # Begitu guru berhasil masuk, jawaban siswa yang tersimpan
                    # di database siap ditampilkan pada menu Jawaban Siswa.
                    sinkronkan_jawaban_database(silent=True)
                    st.rerun()
                else:
                    st.error("Kata sandi salah. Silakan coba kembali.")

        if st.button("Kembali ke Halaman Awal"):
            st.session_state["peran"] = None
            st.rerun()


# ===============================================================================
# HALAMAN ADMIN — sidebar lengkap (5 menu) untuk guru/peneliti
# ===============================================================================

DAFTAR_HALAMAN = {
    "Dashboard": halaman_dashboard,
    "Data": halaman_data,
    "Training & Model": halaman_training_model,
    "Klasifikasi & Evaluasi": halaman_klasifikasi_evaluasi,
    "Laporan / Hasil Akhir": halaman_laporan_akhir,
}


def halaman_admin():
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand">'
            '<div class="judul">Analisis Kecanduan Gadget</div>'
            '<div class="peran">Panel Guru / Administrator</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        halaman_dipilih = st.radio(
            "Menu",
            options=list(DAFTAR_HALAMAN.keys()),
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Alur kerja: Data (Upload, Data Siswa, Preprocessing) → "
                   "Training & Model (Split, Naive Bayes) → "
                   "Klasifikasi & Evaluasi → Laporan.")
        st.divider()
        if st.button("Keluar dari Panel", use_container_width=True):
            st.session_state["peran"] = None
            st.session_state["admin_terautentikasi"] = False
            st.rerun()

    DAFTAR_HALAMAN[halaman_dipilih]()


# ===============================================================================
# HALAMAN AWAL — PILIH PERAN (Siswa atau Guru/Administrator)
# ===============================================================================

def halaman_pilih_peran():
    st.markdown("""
    <div class="app-hero">
        <div class="eyebrow">Sistem Informasi Penelitian</div>
        <h1>Analisis Tingkat Kecanduan Gadget pada Siswa</h1>
        <p>Platform kuesioner dan analisis berbasis metode <b>Naive Bayes</b> untuk membantu pengelolaan data siswa secara terstruktur, aman, dan mudah dipantau oleh guru.</p>
        <span class="hero-badge">SMP Karya Pembangunan Pasirjambu &nbsp;•&nbsp; Naive Bayes</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-strip" style="margin-bottom:18px;">
        <b style="color:#526579;">Pilih akses sistem</b>
        <span style="color:#66778b;"> — pilih akses sesuai kebutuhan Anda.</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("""
        <div class="choice-card">
            <div class="choice-icon">S</div>
            <h3>Kuesioner Siswa</h3>
            <p>Isi kuesioner kecanduan gadget dengan 15 pernyataan. Setelah dikirim, jawaban tersimpan dan dapat dilihat oleh guru/admin.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Mulai Isi Kuesioner", use_container_width=True, type="primary", key="btn_siswa"):
            st.session_state["peran"] = "siswa"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="choice-card">
            <div class="choice-icon">G</div>
            <h3>Akses Guru / Admin</h3>
            <p>Kelola data siswa, lihat jawaban kuesioner yang masuk, proses Naive Bayes, evaluasi hasil klasifikasi, dan laporan penelitian.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Masuk  ", use_container_width=True, key="btn_admin"):
            st.session_state["peran"] = "admin"
            st.rerun()

    st.markdown("""
    <div style="text-align:center;color:#8191a4;font-size:.76rem;margin-top:30px;">
        Jawaban disimpan pada file lokal di komputer yang menjalankan aplikasi.
    </div>
    """, unsafe_allow_html=True)


# ===============================================================================
# GERBANG UTAMA
# ===============================================================================

peran_aktif = st.session_state.get("peran")

if peran_aktif is None:
    halaman_pilih_peran()
elif peran_aktif == "siswa":
    halaman_portal_siswa()
elif peran_aktif == "admin":
    if st.session_state.get("admin_terautentikasi"):
        halaman_admin()
    else:
        halaman_login_admin()

