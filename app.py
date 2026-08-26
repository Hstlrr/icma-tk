import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import networkx as nx
from pyvis.network import Network # type: ignore
import streamlit.components.v1 as components
import tempfile
import os
import re

st.set_page_config(
    page_title="ICMA - Isnad cum Matn Analysis",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for high contrast readability and RTL Arabic typography
st.markdown("""
<style>
    .hadith-arabic-box {
        direction: rtl;
        font-family: 'Amiri', 'Traditional Arabic', 'Scheherazade', serif;
        font-size: 22px;
        line-height: 2.2;
        color: #0f172a !important;
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .hadith-english-box {
        direction: ltr;
        font-size: 16px;
        line-height: 1.7;
        color: #0f172a !important;
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .isnad-matn-block {
        font-family: monospace;
        font-size: 15px;
        background-color: #f1f5f9;
        color: #0f172a !important;
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 10px 0;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# Common Arabic isnad formula stop words
ISNAD_STOP_WORDS = [
    'حدثنا', 'اخبرنا', 'انبانا', 'حدثني', 'اخبرني', 'سمعت', 'سمع', 'قال', 'قالت', 'عن', 
    'ان', 'انه', 'انها', 'رضي', 'الله', 'عنه', 'عنها', 'عنهم', 'صلى', 'عليه', 'وسلم', 
    'النبي', 'رسول', 'بن', 'ابن', 'ابي', 'ابو', 'ام', 'في', 'من', 'ما', 'لا', 'ثم', 'او'
]

# --- Arabic Normalization ---
def normalize_arabic(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652\u06D6-\u06ED]', '', text) # Remove diacritics
    text = re.sub(r'[إأآا]', 'ا', text) # Normalize alefs
    text = re.sub(r'ة', 'ه', text) # Normalize teh marbuta
    text = re.sub(r'[يى]', 'ي', text) # Normalize yaa
    text = re.sub(r'ـ', '', text) # Remove tatweel
    return ' '.join(text.split())

# --- Isnad and Matn Splitting ---
def split_isnad_matn(text_ar):
    if not isinstance(text_ar, str) or not text_ar.strip():
        return "", ""
    text = text_ar.strip()
    
    # 1. Quote markers
    quote_match = re.search(r'‏"‏(.*?)‏"‏', text)
    if quote_match and quote_match.start() > 15:
        return text[:quote_match.start()].strip(), text[quote_match.start():].strip()
        
    # 2. Key transition patterns to matn
    markers = [
        r'(قال\s+سمعت\s+رسول\s+الله)',
        r'(سمعت\s+رسول\s+الله)',
        r'(أن\s+رسول\s+الله)',
        r'(ان\s+رسول\s+الله)',
        r'(قال\s+رسول\s+الله)',
        r'(عن\s+النبي\s+صلى\s+الله\s+عليه\s+وسلم\s+قال)',
        r'(عن\s+النبي\s+صلى\s+الله\s+عليه\s+وسلم)',
        r'(أن\s+النبي\s+صلى\s+الله\s+عليه\s+وسلم)',
        r'(ان\s+النبي\s+صلى\s+الله\s+عليه\s+وسلم)',
        r'(قال\s+سمعته\s+يأمر)',
        r'(قال\s+سمعته\s+يقول)',
        r'(سمعته\s+يقول)',
        r'(سمعت\s+أبا\s+سعيد)',
        r'(سمعت\s+ابا\s+سعيد)',
        r'(أنه\s+سمع)',
        r'(أنه\s+قال)',
        r'(قال\s+قال)',
        r'(فقال\s+رسول\s+الله)',
        r'(أن\s+رسول\s+الله\s+صلى\s+الله\s+عليه\s+وسلم\s+نهى)',
        r'(نهى\s+رسول\s+الله)'
    ]
    for p in markers:
        m = re.search(p, text)
        if m and m.start() > 10:
            return text[:m.start()].strip(), text[m.start():].strip()
            
    # Fallback: estimate from chain words
    words = text.split()
    if len(words) > 12:
        return " ".join(words[:8]), " ".join(words[8:])
    return text, text

# --- Module A: Data Ingestion & Pre-computation ---
@st.cache_data
def load_data():
    hadiths_df = pd.read_csv("all_hadiths_clean.csv", encoding="utf-8")
    rawis_df = pd.read_csv("all_rawis.csv", encoding="utf-8")
    
    # Fill NAs and clean
    hadiths_df['text_ar'] = hadiths_df['text_ar'].fillna('')
    hadiths_df['text_en'] = hadiths_df['text_en'].fillna('')
    hadiths_df['chain_indx'] = hadiths_df['chain_indx'].fillna('')
    hadiths_df['source'] = hadiths_df['source'].fillna('').str.strip()
    hadiths_df['chapter'] = hadiths_df['chapter'].fillna('')
    
    rawis_df['scholar_indx'] = rawis_df['scholar_indx'].astype(str).str.strip()
    rawis_df['name'] = rawis_df['name'].fillna('Unknown')
    rawis_df['grade'] = rawis_df['grade'].fillna('Unknown')
    
    # Pre-normalize full Arabic text
    hadiths_df['text_ar_norm'] = hadiths_df['text_ar'].apply(normalize_arabic)
    
    # Split isnad and matn
    splits = hadiths_df['text_ar'].apply(split_isnad_matn)
    hadiths_df['isnad_ar'] = [s[0] for s in splits]
    hadiths_df['matn_ar'] = [s[1] for s in splits]
    hadiths_df['matn_ar_norm'] = hadiths_df['matn_ar'].apply(normalize_arabic)
    
    # Pre-compute TF-IDF Matrices:
    # 1. Full Arabic Text TF-IDF
    tfidf_full = TfidfVectorizer(ngram_range=(1, 2), max_features=25000, sublinear_tf=True)
    matrix_full = tfidf_full.fit_transform(hadiths_df['text_ar_norm'])
    
    # 2. Matn-only Arabic TF-IDF (Filtered from transmission formula noise)
    tfidf_matn = TfidfVectorizer(ngram_range=(1, 2), max_features=30000, stop_words=ISNAD_STOP_WORDS, sublinear_tf=True)
    matrix_matn = tfidf_matn.fit_transform(hadiths_df['matn_ar_norm'])
    
    # 3. English Translation TF-IDF
    tfidf_en = TfidfVectorizer(ngram_range=(1, 2), max_features=30000, stop_words='english', sublinear_tf=True)
    matrix_en = tfidf_en.fit_transform(hadiths_df['text_en'])
    
    # Fast O(1) dictionary for Rawis
    rawis_dict = rawis_df.set_index('scholar_indx').to_dict(orient='index')
    
    return hadiths_df, rawis_df, rawis_dict, tfidf_full, matrix_full, tfidf_matn, matrix_matn, tfidf_en, matrix_en

try:
    with st.spinner("Loading Hadiths and computing multi-vector TF-IDF indices..."):
        (hadiths_df, rawis_df, rawis_dict,
         tfidf_full, matrix_full,
         tfidf_matn, matrix_matn,
         tfidf_en, matrix_en) = load_data()
except Exception as e:
    st.error(f"Error loading datasets: {e}")
    st.stop()

def get_rawi_info(scholar_indx):
    s_id = str(scholar_indx).strip()
    return rawis_dict.get(s_id, None)

# --- Sidebar: Search & Controls ---
st.sidebar.title("🔍 ICMA Controls")

search_mode = st.sidebar.radio(
    "Search Base Hadith By:",
    ["Text / Keywords / Arabic", "Hadith ID", "Narrator Name"],
    index=0
)

search_query = st.sidebar.text_input("Search Base Hadith", "")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Variant Matching Strategy")
retrieval_strategy = st.sidebar.selectbox(
    "Similarity Metric:",
    [
        "Hybrid ICMA (Matn + English + Narrator Overlap)",
        "Matn-Only Arabic Similarity",
        "Full Text (Isnad + Matn) Similarity",
        "English Translation Similarity"
    ],
    index=0
)

sim_threshold = st.sidebar.slider("Cosine Similarity Threshold", min_value=0.05, max_value=0.95, value=0.20, step=0.05)
top_k = st.sidebar.slider("Max Variants (Top-K)", min_value=1, max_value=50, value=15, step=1)

all_sources = sorted(list(hadiths_df['source'].unique()))
source_filters = st.sidebar.multiselect("Filter Collections", options=all_sources, default=[])

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Graph Styling")
graph_direction = st.sidebar.selectbox("Tree Hierarchy Direction", ["UD (Top-to-Bottom: Prophet at Bottom)", "DU (Bottom-to-Top: Prophet at Top)"], index=0)
hierarchical_dir = "UD" if "UD" in graph_direction else "DU"

# --- Module B: Search Engine ---
def search_hadiths(query, mode):
    query = query.strip()
    if not query:
        return pd.DataFrame()
    
    if mode == "Hadith ID" or (mode != "Narrator Name" and query.isdigit()):
        try:
            q_id = int(query)
            return hadiths_df[hadiths_df['hadith_id'] == q_id]
        except:
            return pd.DataFrame()
            
    elif mode == "Narrator Name":
        matched_rawis = rawis_df[rawis_df['name'].str.contains(query, case=False, na=False)]
        if matched_rawis.empty:
            return pd.DataFrame()
        matched_ids = set(matched_rawis['scholar_indx'].tolist())
        def has_rawi(chain_str):
            if not chain_str:
                return False
            tokens = {t.strip() for t in chain_str.split(',') if t.strip()}
            return bool(tokens & matched_ids)
        mask = hadiths_df['chain_indx'].apply(has_rawi)
        return hadiths_df[mask]
        
    else:
        norm_q = normalize_arabic(query)
        # Substring matching on normalized Arabic or English
        mask_ar = hadiths_df['text_ar_norm'].str.contains(norm_q, regex=False, na=False)
        mask_en = hadiths_df['text_en'].str.contains(query, case=False, regex=False, na=False)
        results = hadiths_df[mask_ar | mask_en]
        
        # Fallback to TF-IDF if query is a long sentence and no exact substring matches
        if results.empty and len(norm_q) > 3:
            q_vec = tfidf_full.transform([norm_q])
            scores = cosine_similarity(q_vec, matrix_full).flatten()
            best_idx = np.where(scores >= 0.15)[0]
            if len(best_idx) > 0:
                best_sorted = best_idx[np.argsort(scores[best_idx])[::-1]][:25]
                results = hadiths_df.iloc[best_sorted]
                
        return results

selected_hadith = None
if search_query:
    search_results = search_hadiths(search_query, search_mode)
    if not search_results.empty:
        st.sidebar.success(f"Found {len(search_results)} candidate Hadith(s).")
        
        options_dict = {}
        for _, row in search_results.head(100).iterrows():
            h_id = row['hadith_id']
            preview = row['text_ar'][:60].strip()
            src = row['source']
            options_dict[h_id] = f"[{src} #{h_id}] {preview}..."
            
        selected_id = st.sidebar.selectbox(
            "Select Base Hadith:",
            options=list(options_dict.keys()),
            format_func=lambda x: options_dict[x]
        )
        selected_hadith = hadiths_df[hadiths_df['hadith_id'] == selected_id].iloc[0]
    else:
        st.sidebar.warning("No matching Hadiths found. Try a different keyword or Hadith ID.")

# --- Variant Retrieval Computation ---
def get_hadith_variants(base_row, strategy, threshold, max_k, sources_filter):
    base_idx = base_row.name
    
    if strategy == "Matn-Only Arabic Similarity":
        base_vec = matrix_matn[base_idx]
        sim_scores = cosine_similarity(base_vec, matrix_matn).flatten()
    elif strategy == "Full Text (Isnad + Matn) Similarity":
        base_vec = matrix_full[base_idx]
        sim_scores = cosine_similarity(base_vec, matrix_full).flatten()
    elif strategy == "English Translation Similarity":
        base_vec = matrix_en[base_idx]
        sim_scores = cosine_similarity(base_vec, matrix_en).flatten()
    else:
        # Hybrid ICMA: 0.50 Matn + 0.35 English + 0.15 Narrator Overlap
        matn_sim = cosine_similarity(matrix_matn[base_idx], matrix_matn).flatten()
        en_sim = cosine_similarity(matrix_en[base_idx], matrix_en).flatten()
        
        base_chain = set([x.strip() for x in str(base_row['chain_indx']).split(',') if x.strip()])
        def chain_overlap(c_str):
            if not c_str or not base_chain:
                return 0.0
            cand_chain = set([x.strip() for x in str(c_str).split(',') if x.strip()])
            if not cand_chain:
                return 0.0
            return len(base_chain & cand_chain) / max(len(base_chain), 1)
            
        overlap_scores = np.array([chain_overlap(c) for c in hadiths_df['chain_indx']])
        sim_scores = (0.50 * matn_sim) + (0.35 * en_sim) + (0.15 * overlap_scores)
        
    ranked_indices = np.argsort(sim_scores)[::-1]
    
    base_narrators = set([x.strip() for x in str(base_row['chain_indx']).split(',') if x.strip()])
    
    variants = []
    for idx in ranked_indices:
        if idx == base_idx:
            continue
        score = float(sim_scores[idx])
        if score < threshold:
            break
            
        cand_row = hadiths_df.iloc[idx]
        if sources_filter and cand_row['source'] not in sources_filter:
            continue
            
        cand_narrators = set([x.strip() for x in str(cand_row['chain_indx']).split(',') if x.strip()])
        common_narrators = base_narrators & cand_narrators
        
        variants.append({
            'idx': idx,
            'sim': score,
            'data': cand_row,
            'common_rawis_count': len(common_narrators),
            'common_rawis': list(common_narrators)
        })
        if len(variants) >= max_k:
            break
            
    return variants

# --- Module C: Word-Level Text Diff Engine ---
def compute_word_diff(text_a, text_b, is_arabic=True):
    words_a = text_a.split()
    words_b = text_b.split()
    
    matcher = difflib.SequenceMatcher(None, words_a, words_b)
    html_out = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            html_out.append(f"<span style='color: #0f172a;'>{' '.join(words_a[i1:i2])}</span>")
        elif tag == 'delete':
            deleted = " ".join(words_a[i1:i2])
            html_out.append(f"<span style='background-color: #fee2e2; color: #991b1b; text-decoration: line-through; padding: 2px 5px; border-radius: 4px; font-weight: bold;'>{deleted}</span>")
        elif tag == 'insert':
            inserted = " ".join(words_b[j1:j2])
            html_out.append(f"<span style='background-color: #dcfce7; color: #166534; font-weight: bold; padding: 2px 5px; border-radius: 4px;'>{inserted}</span>")
        elif tag == 'replace':
            deleted = " ".join(words_a[i1:i2])
            inserted = " ".join(words_b[j1:j2])
            html_out.append(f"<span style='background-color: #fee2e2; color: #991b1b; text-decoration: line-through; padding: 2px 5px; border-radius: 4px;'>{deleted}</span> <span style='background-color: #dcfce7; color: #166534; font-weight: bold; padding: 2px 5px; border-radius: 4px;'>{inserted}</span>")
            
    direction = "rtl" if is_arabic else "ltr"
    font_size = "20px" if is_arabic else "16px"
    line_height = "2.2" if is_arabic else "1.7"
    
    return f"<div dir='{direction}' style='font-size: {font_size}; line-height: {line_height}; color: #0f172a; padding: 14px; border: 1px solid #cbd5e1; border-radius: 8px; background-color: #f8fafc;'>{' '.join(html_out)}</div>"

# --- Module D: Composite Isnad Network & Tree Visualization ---
def get_node_color_and_role(grade_str, in_deg, out_deg, is_cl, is_pcl):
    if is_cl:
        return "#E11D48", "Common Link (CL)" # Rose / Red for Common Link
    if is_pcl:
        return "#EA580C", "Partial Common Link (PCL)" # Orange for Partial Common Link
        
    g_lower = str(grade_str).lower()
    if 'rasool' in g_lower or 'prophet' in g_lower:
        return "#16A34A", "Prophet (ص)" # Green
    if 'comp.(ra)' in g_lower or 'sahaba' in g_lower or '1st generation' in g_lower:
        return "#2563EB", "Companion (RA)" # Blue
    if 'thiqa' in g_lower or 'trustworthy' in g_lower:
        return "#0891B2", "Thiqa (Trustworthy)" # Cyan
    if 'daif' in g_lower or 'weak' in g_lower or 'munkar' in g_lower:
        return "#DC2626", "Da'if / Weak" # Red
    return "#64748B", "Narrator" # Slate Grey

def build_composite_isnad_graph(base_hadith, variants, hierarchy_dir="UD"):
    G = nx.DiGraph()
    all_hadith_records = [base_hadith] + [v['data'] for v in variants]
    
    chains = []
    for h in all_hadith_records:
        chain_str = str(h.get('chain_indx', ''))
        if not chain_str or chain_str == 'nan':
            continue
        ids = [t.strip() for t in chain_str.split(',') if t.strip()]
        if ids:
            chains.append(ids)
            for i in range(len(ids) - 1):
                src = ids[i]
                dst = ids[i+1]
                G.add_edge(src, dst)
                
    if len(G.nodes) == 0:
        return G, None, None
        
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    
    node_chain_counts = {n: 0 for n in G.nodes()}
    for chain in chains:
        for n in set(chain):
            if n in node_chain_counts:
                node_chain_counts[n] += 1
                
    cl_node = None
    pcl_nodes = []
    
    sorted_by_convergence = sorted(
        G.nodes(),
        key=lambda n: (in_degrees.get(n, 0) + node_chain_counts.get(n, 0)),
        reverse=True
    )
    
    if len(sorted_by_convergence) > 0 and len(chains) > 1:
        for cand in sorted_by_convergence:
            info = get_rawi_info(cand)
            grade = info.get('grade', '') if info else ''
            # Exclude the Prophet from being designated as intermediate CL
            if 'rasool' not in str(grade).lower() and 'prophet' not in str(grade).lower():
                if cl_node is None:
                    cl_node = cand
                elif in_degrees.get(cand, 0) > 1 or node_chain_counts.get(cand, 0) > 1:
                    pcl_nodes.append(cand)
                    
    for node in G.nodes():
        info = get_rawi_info(node)
        name = info.get('name', f"ID: {node}") if info else f"ID: {node}"
        grade = info.get('grade', 'Unknown') if info else 'Unknown'
        death_h = info.get('death_date_hijri', 'N/A') if info else 'N/A'
        places = info.get('places_of_stay', 'N/A') if info else 'N/A'
        
        is_cl = (node == cl_node)
        is_pcl = (node in pcl_nodes)
        color, role = get_node_color_and_role(grade, in_degrees.get(node, 0), out_degrees.get(node, 0), is_cl, is_pcl)
        
        label_text = f"{name}\n({death_h} AH)" if death_h and str(death_h) != 'nan' and str(death_h) != 'N/A' else name
        tooltip = f"<b>{name}</b><br>Scholar ID: {node}<br>Role: {role}<br>Grade: {grade}<br>Death: {death_h} AH<br>Places: {places}"
        
        G.nodes[node]['label'] = label_text
        G.nodes[node]['title'] = tooltip
        G.nodes[node]['color'] = color
        G.nodes[node]['shape'] = 'box'
        G.nodes[node]['font'] = {'size': 14, 'face': 'sans-serif', 'color': '#ffffff'}
        G.nodes[node]['borderWidth'] = 3 if (is_cl or is_pcl) else 1
        
    return G, cl_node, pcl_nodes

def render_pyvis_network(G, hierarchy_dir="UD"):
    net = Network(height='650px', width='100%', directed=True, layout=True)
    net.from_nx(G)
    
    net.set_options(f'''
    var options = {{
      "layout": {{
        "hierarchical": {{
          "enabled": true,
          "direction": "{hierarchy_dir}",
          "sortMethod": "directed",
          "nodeSpacing": 170,
          "levelSeparation": 130
        }}
      }},
      "physics": {{
        "enabled": false
      }},
      "edges": {{
        "smooth": {{
          "type": "cubicBezier",
          "forceDirection": "vertical",
          "roundness": 0.4
        }},
        "color": {{
          "color": "#94a3b8",
          "highlight": "#2563eb"
        }},
        "arrows": {{
          "to": {{ "enabled": true, "scaleFactor": 0.8 }}
        }}
      }},
      "interaction": {{
        "hover": true,
        "navigationButtons": true,
        "zoomView": true
      }}
    }}
    ''')
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as f:
        path = f.name
        net.save_graph(path)
        
    with open(path, 'r', encoding='utf-8') as f:
        html_data = f.read()
    try:
        os.remove(path)
    except:
        pass
        
    return html_data

# --- Dashboard Header ---
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>📜 Isnad-cum-Matn Analysis (ICMA) Platform</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748B;'>Advanced Digital Humanities Suite for Hadith Variant Diffing, Isnad Reconstruction, and Common Link Detection</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Tabs Setup ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 1. Overview & Chains",
    "🔍 2. Matn Variant Diffing",
    "🌳 3. Unified Isnad Graph",
    "👤 4. Narrator Biographies"
])

if selected_hadith is not None:
    variants = get_hadith_variants(selected_hadith, retrieval_strategy, sim_threshold, top_k, source_filters)
    
    # === TAB 1: OVERVIEW & CHAINS ===
    with tab1:
        st.subheader("📌 Base Hadith Reference")
        c1, c2, c3 = st.columns([2, 2, 1])
        c1.info(f"**Collection:** {selected_hadith['source']}")
        c2.info(f"**Chapter:** {selected_hadith['chapter']}")
        c3.info(f"**Hadith ID:** #{selected_hadith['hadith_id']}")
        
        # Explicit Isnad & Matn Format Display
        isnad_text = selected_hadith['isnad_ar']
        matn_text = selected_hadith['matn_ar']
        
        st.markdown("#### 🧩 Structured Isnad & Matn:")
        isnad_matn_formatted = f'isnad: "{isnad_text}"\nmatn: "{matn_text}"'
        st.code(isnad_matn_formatted, language="yaml")
        
        # Individual Narrators identified in the chain
        chain_ids = [x.strip() for x in str(selected_hadith['chain_indx']).split(',') if x.strip()]
        if chain_ids:
            st.markdown("#### 🔗 Transmission Chain Narrators:")
            narrator_tags = []
            for cid in chain_ids:
                info = get_rawi_info(cid)
                n_name = info['name'] if info else f"ID: {cid}"
                narrator_tags.append(f"**{n_name}** (`{cid}`)")
            st.markdown(" ➔ ".join(narrator_tags))
            
        st.markdown("#### 📖 Full Arabic Text (`text_ar`):")
        st.markdown(f"<div class='hadith-arabic-box'>{selected_hadith['text_ar']}</div>", unsafe_allow_html=True)
        
        st.markdown("#### 🌐 English Translation (`text_en`):")
        st.markdown(f"<div class='hadith-english-box'>{selected_hadith['text_en']}</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader(f"📑 Retrieved Variant Corpus ({len(variants)} matches found using {retrieval_strategy})")
        
        if variants:
            v_table = []
            for i, v in enumerate(variants):
                v_data = v['data']
                v_table.append({
                    "Rank": i + 1,
                    "Similarity": f"{v['sim'] * 100:.1f}%",
                    "Common Narrators": v['common_rawis_count'],
                    "Source Collection": v_data['source'],
                    "Hadith ID": v_data['hadith_id'],
                    "Chapter": v_data['chapter'],
                    "Matn Preview": v_data['matn_ar'][:85] + "..."
                })
            st.dataframe(pd.DataFrame(v_table), use_container_width=True)
        else:
            st.info("No variants matched the current similarity threshold. Try lowering the threshold or changing the matching strategy in the sidebar.")

    # === TAB 2: MATN DIFFING ===
    with tab2:
        st.subheader("🔬 Word-Level Matn Text Diffing (Base vs. Variants)")
        st.caption("🟢 Green = Addition in Variant | 🔴 Red Strikethrough = Omission / Mutation from Base")
        
        if not variants:
            st.warning("No variants available for diff comparison.")
        else:
            diff_scope = st.radio("Diff Target", ["Matn Only (Recommended for ICMA)", "Full Text (Isnad + Matn)", "English Translation"], horizontal=True)
            
            for idx, v in enumerate(variants):
                v_data = v['data']
                sim_pct = f"{v['sim'] * 100:.1f}%"
                
                with st.expander(f"Variant #{idx+1}: {v_data['source']} [Hadith #{v_data['hadith_id']}] — Similarity: {sim_pct} ({v['common_rawis_count']} shared narrators)", expanded=(idx == 0)):
                    st.markdown(f"**Chapter:** {v_data['chapter']}")
                    
                    if "Matn Only" in diff_scope:
                        st.markdown("**Matn Diff (Arabic):**")
                        diff_html = compute_word_diff(selected_hadith['matn_ar'], v_data['matn_ar'], is_arabic=True)
                        st.markdown(diff_html, unsafe_allow_html=True)
                    elif "Full Text" in diff_scope:
                        st.markdown("**Full Text Diff (Arabic):**")
                        diff_html = compute_word_diff(selected_hadith['text_ar'], v_data['text_ar'], is_arabic=True)
                        st.markdown(diff_html, unsafe_allow_html=True)
                    else:
                        st.markdown("**English Translation Diff:**")
                        diff_html = compute_word_diff(selected_hadith['text_en'], v_data['text_en'], is_arabic=False)
                        st.markdown(diff_html, unsafe_allow_html=True)

    # === TAB 3: UNIFIED ISNAD GRAPH ===
    with tab3:
        st.subheader("🕸️ Unified Transmission Tree (DAG)")
        st.caption("Interactive graph generated from the merged chains of the Base Hadith and all retrieved variants.")
        
        G, cl_node, pcl_nodes = build_composite_isnad_graph(selected_hadith, variants, hierarchy_dir=hierarchical_dir)
        
        if len(G.nodes) > 0:
            col_l1, col_l2, col_l3, col_l4, col_l5 = st.columns(5)
            col_l1.markdown("🔴 **Common Link (CL)**")
            col_l2.markdown("🟠 **Partial Common Link (PCL)**")
            col_l3.markdown("🟢 **Prophet (ص)**")
            col_l4.markdown("🔵 **Companions (RA)**")
            col_l5.markdown("⚪ **Other Narrators**")
            
            if cl_node:
                cl_info = get_rawi_info(cl_node)
                cl_name = cl_info['name'] if cl_info else f"ID: {cl_node}"
                st.success(f"🎯 **Identified Potential Common Link (CL):** {cl_name} (Scholar ID: `{cl_node}`)")
            if pcl_nodes:
                pcl_names = [get_rawi_info(n)['name'] if get_rawi_info(n) else f"ID: {n}" for n in pcl_nodes[:3]]
                st.info(f"⚡ **Partial Common Link(s) (PCL):** {', '.join(pcl_names)}")
                
            graph_html = render_pyvis_network(G, hierarchy_dir=hierarchical_dir)
            components.html(graph_html, height=680, scrolling=True)
        else:
            st.warning("No transmission chain data (`chain_indx`) available for the selected hadiths.")

    # === TAB 4: NARRATOR BIOGRAPHIES ===
    with tab4:
        st.subheader("👤 Narrator Biodata & Relationship Inspector")
        
        all_chain_nodes = []
        for h in [selected_hadith] + [v['data'] for v in variants]:
            c_str = str(h.get('chain_indx', ''))
            if c_str and c_str != 'nan':
                for nid in c_str.split(','):
                    nid_clean = nid.strip()
                    if nid_clean and nid_clean not in all_chain_nodes:
                        all_chain_nodes.append(nid_clean)
                        
        if all_chain_nodes:
            narrator_choices = {}
            for nid in all_chain_nodes:
                info = get_rawi_info(nid)
                nname = info.get('name', f"Unknown Narrator ({nid})") if info else f"Unknown Narrator ({nid})"
                narrator_choices[nid] = f"{nname} [ID: {nid}]"
                
            selected_narrator_id = st.selectbox(
                "Select Narrator to Inspect:",
                options=all_chain_nodes,
                format_func=lambda x: narrator_choices[x]
            )
            
            if selected_narrator_id:
                rawi = get_rawi_info(selected_narrator_id)
                if rawi:
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.markdown(f"### {rawi.get('name', 'N/A')}")
                        st.markdown(f"**Scholar Index:** `{selected_narrator_id}`")
                        st.markdown(f"**Reliability Grade:** `{rawi.get('grade', 'N/A')}`")
                        st.markdown(f"**Death Date (Hijri):** {rawi.get('death_date_hijri', 'N/A')} AH")
                        st.markdown(f"**Death Date (Gregorian):** {rawi.get('death_date_gregorian', 'N/A')} CE")
                        st.markdown(f"**Birth Place / Dates:** {rawi.get('birth_date_place', 'N/A')}")
                        st.markdown(f"**Places of Stay:** {rawi.get('places_of_stay', 'N/A')}")
                        
                    with col2:
                        st.markdown("#### 📚 Scholarly Context & Networks")
                        st.markdown(f"**Areas of Interest:** {rawi.get('area_of_interest', 'N/A')}")
                        st.markdown(f"**Tags:** {rawi.get('tags', 'N/A')}")
                        
                        teachers = str(rawi.get('teachers', 'N/A'))
                        students = str(rawi.get('students', 'N/A'))
                        
                        with st.expander("🎓 Direct Teachers", expanded=False):
                            st.write(teachers)
                        with st.expander("👥 Direct Students", expanded=False):
                            st.write(students)
                else:
                    st.warning(f"No detailed biographical entry found for Scholar ID {selected_narrator_id}.")
        else:
            st.info("No narrator IDs found in the current transmission chains.")

else:
    with tab1:
        st.info("👈 Please enter a keyword, Hadith ID, or narrator name in the sidebar to begin analysis.")
        st.markdown("""
        ### Example Searches:
        - **Hadith ID:** `52336`, `1`, `50241`
        - **Keywords / Text:** `الصرف`, `إنما الأعمال بالنيات`, `أحمد بن عبدة`
        - **Narrators:** `حَمَّادُ بْنُ زَيْدٍ`, `ابن عباس`, `سفيان`
        """)
