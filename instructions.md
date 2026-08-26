You are an expert Python developer and Islamic Digital Humanities specialist. Build a production-ready Streamlit web application for **Isnad-cum-Matn Analysis (ICMA)** of Hadith literature using the provided datasets: `all_hadiths_clean.csv` and `all_rawis.csv`.

---

### 1. Dataset Architecture & Schema Understanding

#### `all_hadiths_clean.csv`
- `id`, `hadith_id`: Unique identifiers.
- `source`: Hadith collection name (e.g., Sahih Bukhari).
- `chapter_no`, `hadith_no`, `chapter`: Reference metadata.
- `chain_indx`: Comma-separated list of scholar IDs representing the transmission chain from compiler down to the original speaker (e.g., `'30418, 20005, 11062, 11213, 11042, 3'`).
- `text_ar`: Full Arabic text (containing isnad formula + matn).
- `text_en`: English translation.

#### `all_rawis.csv`
- `scholar_indx`: Matches the IDs in `chain_indx`.
- `name`: Full name of narrator (English + Arabic).
- `grade`: Reliability evaluation / generation (e.g., `Comp.(RA) [1st Generation]`, `Thiqa`).
- `death_date_hijri`, `birth_date_hijri`: Chronological markers.
- `places_of_stay`, `area_of_interest`, `teachers_inds`, `students_inds`.

---

### 2. Core Functional Modules

#### Module A: Data Ingestion & Pre-computation
- Load datasets using `@st.cache_data`.
- Parse `chain_indx` into integer lists and map each ID to `all_rawis.csv` records.
- Build a lightweight text-similarity index over `text_ar` and `text_en` using `sklearn.feature_extraction.text.TfidfVectorizer` (with Arabic char/word n-grams and English word TF-IDF) and cosine similarity for rapid variant retrieval.

#### Module B: Variant Retrieval Engine
- Allow the researcher to:
  1. Select a base hadith by searching book/hadith number, keyword, or narrator name.
  2. Adjust a **Similarity Threshold Slider** (e.g., 0.30 to 0.95) or specify Top-$K$ matches.
  3. Filter variants by source collections.
- Output a list of variant hadiths alongside their similarity scores and common narrators.

#### Module C: Matn Comparison & Word-Level Text Diff
- Provide side-by-side or inline word diffing between the base Hadith and selected variants.
- Implement HTML-rendered word-level diffs using Python's `difflib.ndiff` (green highlighting for additions, red strikethrough for omissions/mutations).
- Support bidirectional diffs (Arabic and English).

#### Module D: Composite Isnad Network & Tree Visualization
- Merge the transmission chains (`chain_indx`) of all retrieved variants into a single unified Directed Acyclic Graph (DAG) using `graphviz` or `pyvis` (or `networkx` + `st.graphviz_chart`).
- **Graph Layout & Styling Rules**:
  - **Hierarchical Layout**: Place earlier generations (Prophet/Companions) at the top or bottom consistently based on chronology (`death_date_hijri`).
  - **Node Labels**: Display narrator name, death year (AH), and grade.
  - **Node Colors**: Distinct color coding based on reliability (`grade`) or generation.
  - **Analytical Highlighting**: Automatically detect and highlight potential **Common Links (CL)** and **Partial Common Links (PCL)** (nodes with the highest in-degree / branching factor across variant chains).

#### Module E: Narrator Biodata Inspector
- Clicking or selecting any narrator in the chain displays a sidebar/modal profile:
  - Name, Hijri/Gregorian death year, places lived/stayed.
  - Grade / reliability ranking.
  - Direct student-teacher relationships within the selected corpus.

---

### 3. UI/UX Layout Specification

1. **Sidebar**:
   - Hadith selector (Search by ID, Collection, or Text Search).
   - Variant matching parameters (Similarity Threshold, Top-K, Source filters).
   - Graph styling controls (Hierarchical direction, color schemes).

2. **Main Dashboard (Tabs)**:
   - **Tab 1: Overview & Chains**: Base hadith details, variant list, metadata summary table.
   - **Tab 2: Matn Variant Diffing**: Interactive visual text comparison view.
   - **Tab 3: Unified Isnad Graph**: Interactive zoomable/pannable ICMA transmission tree showing convergence at the Common Link.
   - **Tab 4: Narrator Biographies**: Detailed narrator breakdown table and relationship matrix.

---

### 4. Technical Requirements & Deliverables

- Write a clean, modular, fully functional `app.py` script.
- Ensure all edge cases are handled (missing IDs in `chain_indx`, null dates, malformed chains).
- Provide a `requirements.txt` containing: `streamlit`, `pandas`, `scikit-learn`, `graphviz`, `pyvis`.
- Include instructions for running the app locally.