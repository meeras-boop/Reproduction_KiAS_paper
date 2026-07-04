"""
KiAS: Knowledge-Infused Abstractive Summarization
Complete Streamlit Application for Reproduction and Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import re
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ============================================================
# PHQ-9 LEXICON
# ============================================================

PHQ9_LEXICON = {
    "S1_anhedonia": ["lost interest", "no pleasure", "don't enjoy", "nothing fun", 
                     "not interested", "can't enjoy", "no enjoyment", "not caring"],
    "S2_depressed_mood": ["feeling down", "sad", "hopeless", "depressed", "worthless", 
                          "miserable", "blue", "low", "empty", "numb", "crying"],
    "S3_sleep": ["trouble sleeping", "insomnia", "wake up", "can't sleep", 
                 "sleep issues", "tossing", "turning", "nightmares", "sleep too much"],
    "S4_fatigue": ["no energy", "tired", "fatigue", "exhausted", "lethargic", 
                   "drained", "worn out", "run down", "low energy"],
    "S5_appetite": ["eating too much", "no appetite", "lost weight", "eating less", 
                    "overeating", "not eating", "weight gain", "weight loss"],
    "S6_worthlessness": ["worthless", "guilty", "failure", "useless", "burden", 
                         "inadequate", "not good enough"],
    "S7_concentration": ["can't focus", "distracted", "poor concentration", "can't think", 
                         "forgetful", "brain fog", "mind racing"],
    "S8_psychomotor": ["restless", "slowed down", "agitated", "fidgety", 
                       "can't sit still", "moving slowly"],
    "S9_suicidal": ["want to die", "self-harm", "kill myself", "end it all", 
                    "suicide", "death wish", "hurt myself"]
}

# ============================================================
# CORE KIAS CLASSES
# ============================================================

class DataPreprocessor:
    def __init__(self, lexicon=PHQ9_LEXICON):
        self.lexicon = lexicon
        self.all_phq9_terms = []
        for category, terms in self.lexicon.items():
            self.all_phq9_terms.extend(terms)
    
    def has_phq9_term(self, text: str) -> bool:
        text_lower = text.lower()
        for term in self.all_phq9_terms:
            if term.lower() in text_lower:
                return True
        return False
    
    def get_matching_phq9_terms(self, text: str) -> List[str]:
        text_lower = text.lower()
        matches = []
        for term in self.all_phq9_terms:
            if term.lower() in text_lower:
                matches.append(term)
        return matches
    
    def get_phq9_category(self, term: str) -> str:
        for category, terms in self.lexicon.items():
            if term in terms:
                return category
        return None
    
    def prune_conversation(self, transcript: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        pruned = []
        for speaker, utterance in transcript:
            if self.has_phq9_term(utterance):
                pruned.append((speaker, utterance))
        return pruned
    
    def convert_to_statements(self, transcript: List[Tuple[str, str]]) -> List[str]:
        statements = []
        i = 0
        while i < len(transcript):
            if i + 1 < len(transcript):
                q = transcript[i][1]
                a = transcript[i+1][1]
                statement = f"Participant was asked '{q}', participant said '{a}'"
                statements.append(statement)
                i += 2
            else:
                i += 1
        return statements
    
    def extract_phq9_signals(self, statements: List[str]) -> Dict[str, List[str]]:
        signals = defaultdict(list)
        for stmt in statements:
            matches = self.get_matching_phq9_terms(stmt)
            for match in matches:
                category = self.get_phq9_category(match)
                if category:
                    signals[category].append(stmt)
        return dict(signals)

class WordSemanticScorer:
    def __init__(self, lexicon=PHQ9_LEXICON):
        self.lexicon = lexicon
        self.all_terms = []
        for category, terms in self.lexicon.items():
            self.all_terms.extend(terms)
        
        self.word_vectors = self._build_semantic_embeddings()
    
    def _build_semantic_embeddings(self):
        embeddings = {}
        for i, term in enumerate(self.all_terms):
            embedding = np.zeros(len(self.all_terms))
            embedding[i] = 1.0
            embeddings[term] = embedding
        
        clinical_terms = ['tired', 'sleep', 'feel', 'said', 'asked', 'diagnosed',
                         'depression', 'anxiety', 'stress', 'help', 'support',
                         'wake', 'night', 'eating', 'weight', 'focus', 'hopeless',
                         'worthless', 'suicidal', 'self-harm', 'energy', 'exhausted']
        for i, word in enumerate(clinical_terms):
            if word not in embeddings:
                embedding = np.random.rand(len(self.all_terms)) * 0.1
                embeddings[word] = embedding
        return embeddings
    
    def get_word_similarity(self, word1: str, word2: str) -> float:
        word1 = word1.lower()
        word2 = word2.lower()
        
        v1 = self.word_vectors.get(word1, None)
        v2 = self.word_vectors.get(word2, None)
        
        if v1 is None:
            for term in self.all_terms:
                if term in word1:
                    v1 = self.word_vectors.get(term, np.zeros(len(self.all_terms)))
                    break
            if v1 is None:
                v1 = np.zeros(len(self.all_terms))
        
        if v2 is None:
            for term in self.all_terms:
                if term in word2:
                    v2 = self.word_vectors.get(term, np.zeros(len(self.all_terms)))
                    break
            if v2 is None:
                v2 = np.zeros(len(self.all_terms))
        
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0.0
        
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    def calculate_wss(self, statement: str) -> float:
        words = re.findall(r'\w+', statement.lower())
        if not words:
            return 0.0
        
        c_values = []
        for word in words:
            max_sim = 0.0
            for term in self.all_terms:
                sim = self.get_word_similarity(word, term)
                if sim > max_sim:
                    max_sim = sim
            c_values.append(max_sim)
        
        N = len(words)
        if N == 0 or sum(c_values) == 0:
            return 0.0
        
        avg_c = sum(c_values) / N
        wss_values = [(c / avg_c) ** 2 for c in c_values]
        return np.mean(wss_values)

class WordGraphBuilder:
    def __init__(self, window_size: int = 7):
        self.window_size = window_size
    
    def build_word_graph(self, statements: List[str]) -> Dict[str, Dict[str, int]]:
        graph = defaultdict(lambda: defaultdict(int))
        for statement in statements:
            words = re.findall(r'\w+', statement.lower())
            for i in range(len(words) - 1):
                graph[words[i]][words[i+1]] += 1
        return dict(graph)
    
    def textrank(self, graph: Dict[str, Dict[str, int]], 
                 iterations: int = 30, d: float = 0.78) -> Dict[str, float]:
        if not graph:
            return {}
        
        scores = {node: 1.0 for node in graph}
        for _ in range(iterations):
            new_scores = {}
            for node in graph:
                score = (1 - d)
                for neighbor, weight in graph[node].items():
                    if neighbor in scores:
                        out_degree = len(graph.get(neighbor, {}))
                        if out_degree > 0:
                            score += d * scores[neighbor] / out_degree
                new_scores[node] = score
            
            total = sum(new_scores.values())
            if total > 0:
                new_scores = {k: v / total for k, v in new_scores.items()}
            scores = new_scores
        return scores
    
    def compute_word_importance(self, statements: List[str]) -> Dict[str, float]:
        if not statements:
            return {}
        graph = self.build_word_graph(statements)
        return self.textrank(graph)
    
    def get_sentence_informativeness(self, statement: str, 
                                     word_importance: Dict[str, float]) -> float:
        words = re.findall(r'\w+', statement.lower())
        if not words or not word_importance:
            return 0.0
        total_score = sum(word_importance.get(word, 0.0) for word in words)
        return total_score / len(words) if words else 0.0

class KiASSummarizer:
    def __init__(self, window_size: int = 7):
        self.preprocessor = DataPreprocessor()
        self.wss_scorer = WordSemanticScorer()
        self.graph_builder = WordGraphBuilder(window_size=window_size)
        self.window_size = window_size
    
    def summarize(self, transcript: List[Tuple[str, str]], 
                  max_sentences: int = 7) -> Dict[str, Any]:
        results = {"steps": {}, "summary": []}
        
        pruned = self.preprocessor.prune_conversation(transcript)
        results["steps"]["pruned"] = {
            "original_count": len(transcript),
            "pruned_count": len(pruned),
            "pruned_transcript": pruned
        }
        
        if not pruned:
            results["summary"] = ["No clinically relevant information found."]
            return results
        
        statements = self.preprocessor.convert_to_statements(pruned)
        results["steps"]["statements"] = statements
        
        if not statements:
            results["summary"] = ["No Q&A pairs found."]
            return results
        
        wss_scores = {stmt: self.wss_scorer.calculate_wss(stmt) for stmt in statements}
        results["steps"]["wss_scores"] = wss_scores
        
        word_importance = self.graph_builder.compute_word_importance(statements)
        informativeness = {
            stmt: self.graph_builder.get_sentence_informativeness(stmt, word_importance)
            for stmt in statements
        }
        results["steps"]["informativeness"] = informativeness
        
        linguistic_quality = {}
        for stmt in statements:
            words = re.findall(r'\w+', stmt.lower())
            if len(words) >= 3:
                trigram_prob = 0.5 * (sum(1 for term in self.wss_scorer.all_terms 
                                         if term in stmt.lower()) / len(words))
            else:
                trigram_prob = 0.1
            linguistic_quality[stmt] = trigram_prob + wss_scores.get(stmt, 0)
        results["steps"]["linguistic_quality"] = linguistic_quality
        
        scored_statements = []
        for stmt in statements:
            i_score = informativeness.get(stmt, 0.0)
            q_score = linguistic_quality.get(stmt, 0.0)
            word_count = len(re.findall(r'\w+', stmt))
            
            if word_count > 0:
                combined_score = (i_score * q_score) / word_count
            else:
                combined_score = 0.0
            
            if self.preprocessor.has_phq9_term(stmt):
                combined_score *= 1.5
            
            scored_statements.append((stmt, combined_score))
        
        scored_statements.sort(key=lambda x: x[1], reverse=True)
        summary = [stmt for stmt, score in scored_statements[:max_sentences]]
        results["summary"] = summary if summary else ["No meaningful summary generated."]
        
        signals = self.preprocessor.extract_phq9_signals(statements)
        results["steps"]["phq9_signals"] = signals
        
        return results

class Evaluator:
    @staticmethod
    def compute_jensen_shannon_divergence(summary: List[str], reference: List[str]) -> float:
        if not summary or not reference:
            return 1.0
        
        def get_word_dist(texts):
            words = []
            for text in texts:
                words.extend(re.findall(r'\w+', text.lower()))
            return Counter(words)
        
        summary_dist = get_word_dist(summary)
        reference_dist = get_word_dist(reference)
        
        all_words = set(summary_dist.keys()) | set(reference_dist.keys())
        if not all_words:
            return 1.0
        
        p = np.array([summary_dist.get(word, 0) for word in all_words])
        q = np.array([reference_dist.get(word, 0) for word in all_words])
        
        p_sum = p.sum()
        q_sum = q.sum()
        if p_sum == 0 or q_sum == 0:
            return 1.0
        
        p = p / p_sum
        q = q / q_sum
        
        m = (p + q) / 2
        kl_pm = np.sum(p * np.log((p + 1e-10) / (m + 1e-10)))
        kl_qm = np.sum(q * np.log((q + 1e-10) / (m + 1e-10)))
        return (kl_pm + kl_qm) / 2
    
    @staticmethod
    def compute_contextual_similarity(summary: List[str], reference: List[str]) -> float:
        if not summary or not reference:
            return 0.0
        
        def get_word_set(texts):
            words = set()
            for text in texts:
                words.update(re.findall(r'\w+', text.lower()))
            return words
        
        summary_words = get_word_set(summary)
        reference_words = get_word_set(reference)
        
        if not summary_words or not reference_words:
            return 0.0
        
        intersection = len(summary_words & reference_words)
        union = len(summary_words | reference_words)
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def compute_thematic_overlap(summary: List[str], reference: List[str]) -> float:
        if not summary or not reference:
            return 0.0
        
        def get_topics(texts):
            topics = set()
            theme_words = {
                'depression': ['depressed', 'depression', 'hopeless', 'worthless', 'down', 'sad'],
                'anxiety': ['anxious', 'anxiety', 'worry', 'nervous', 'stress'],
                'sleep': ['sleep', 'insomnia', 'wake', 'night', 'tossing'],
                'energy': ['energy', 'tired', 'fatigue', 'exhausted'],
                'appetite': ['appetite', 'eating', 'weight', 'hunger'],
                'suicide': ['suicide', 'self-harm', 'kill', 'death', 'end']
            }
            for text in texts:
                text_lower = text.lower()
                for theme, keywords in theme_words.items():
                    if any(kw in text_lower for kw in keywords):
                        topics.add(theme)
            return topics
        
        summary_topics = get_topics(summary)
        reference_topics = get_topics(reference)
        
        if not reference_topics:
            return 0.0
        
        intersection = len(summary_topics & reference_topics)
        return intersection / len(reference_topics)
    
    @staticmethod
    def compute_rouge(summary: List[str], reference: List[str], n: int = 2) -> Dict[str, float]:
        if not summary or not reference:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        def get_ngrams(texts, n):
            ngrams = Counter()
            for text in texts:
                words = re.findall(r'\w+', text.lower())
                for i in range(len(words) - n + 1):
                    ngram = ' '.join(words[i:i+n])
                    ngrams[ngram] += 1
            return ngrams
        
        summary_ngrams = get_ngrams(summary, n)
        reference_ngrams = get_ngrams(reference, n)
        
        if not summary_ngrams or not reference_ngrams:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        overlap = sum(min(summary_ngrams[ng], reference_ngrams[ng]) 
                     for ng in set(summary_ngrams) & set(reference_ngrams))
        
        summary_total = sum(summary_ngrams.values())
        reference_total = sum(reference_ngrams.values())
        
        if summary_total == 0 or reference_total == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        precision = overlap / summary_total
        recall = overlap / reference_total
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return {"precision": precision, "recall": recall, "f1": f1}

# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="KiAS - Clinical Interview Summarizer",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #2c3e50; }
    .sub-header { font-size: 1.5rem; font-weight: bold; color: #34495e; }
    .metric-card { background: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 4px solid #3498db; }
    .highlight { background: #fff3cd; padding: 10px; border-radius: 5px; border-left: 4px solid #ffc107; }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🧠 KiAS: Knowledge-Infused Abstractive Summarization</p>', unsafe_allow_html=True)
st.markdown("*Reproduction of Manas et al., JMIR Mental Health 2021*")

# Load data
@st.cache_data
def load_data():
    try:
        with open('dummy_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("❌ dummy_data.json not found! Please make sure the file is in the same directory.")
        return None

data = load_data()
if data is None:
    st.stop()

# Sidebar
st.sidebar.title("⚙️ Controls")
selected_patient = st.sidebar.selectbox(
    "Select Patient",
    list(data.keys()),
    format_func=lambda x: f"{data[x]['patient_id']} - {data[x]['metadata']['diagnosis']}"
)

max_sentences = st.sidebar.slider("Max Summary Sentences", 3, 10, 7)
window_size = st.sidebar.slider("Window Size (Q&A pairs)", 3, 10, 7)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Paper Metrics")
st.sidebar.info("""
- **JSD**: Lower is better
- **Similarity**: Higher is better  
- **Thematic Overlap**: Higher is better
- **ROUGE-2 F1**: Higher is better
""")

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📝 Patient Data", 
    "📊 Summarization", 
    "📈 Evaluation", 
    "📉 Comparative Charts",
    "🔬 PHQ-9 Analysis"
])

# ===================== TAB 1: Patient Data =====================
with tab1:
    st.markdown('<p class="sub-header">📝 Patient Interview Data</p>', unsafe_allow_html=True)
    
    patient_data = data[selected_patient]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Patient ID", patient_data['patient_id'])
    with col2:
        st.metric("Diagnosis", patient_data['metadata']['diagnosis'])
    with col3:
        st.metric("PHQ-8 Score", f"{patient_data['metadata']['phq8_score']}/24")
    with col4:
        st.metric("Gender", patient_data['metadata']['gender'])
    
    # Display transcript
    st.markdown("### 📄 Full Transcript")
    df = pd.DataFrame(patient_data['transcript'], columns=["Speaker", "Utterance"])
    st.dataframe(df, use_container_width=True)
    
    # Edit option
    with st.expander("✏️ Edit Transcript (Dynamic Update)"):
        st.warning("⚠️ Editing the transcript will trigger re-summarization")
        edited_transcript = []
        for i, (speaker, utterance) in enumerate(patient_data['transcript']):
            col1, col2 = st.columns([1, 4])
            with col1:
                speaker_choice = st.selectbox(
                    f"Speaker {i+1}", 
                    ["Ellie", "Patient"], 
                    index=0 if speaker == "Ellie" else 1,
                    key=f"speaker_{i}"
                )
            with col2:
                new_utterance = st.text_area(
                    f"Utterance {i+1}",
                    value=utterance,
                    key=f"utterance_{i}",
                    height=50
                )
            edited_transcript.append([speaker_choice, new_utterance])
        
        if st.button("🔄 Update Transcript & Re-summarize"):
            patient_data['transcript'] = edited_transcript
            st.success("✅ Transcript updated! Go to the Summarization tab to see results.")

# ===================== TAB 2: Summarization =====================
with tab2:
    st.markdown('<p class="sub-header">📊 Knowledge-Infused Summarization</p>', unsafe_allow_html=True)
    
    # Run summarization
    with st.spinner("🔬 Generating KiAS summary..."):
        summarizer = KiASSummarizer(window_size=window_size)
        result = summarizer.summarize(patient_data['transcript'], max_sentences=max_sentences)
    
    # Display summary
    st.markdown("### 📋 Generated Summary")
    for i, sentence in enumerate(result['summary'], 1):
        st.markdown(f"**{i}.** {sentence}")
    
    # Analysis metrics
    st.markdown("### 📊 Analysis Breakdown")
    
    col1, col2, col3 = st.columns(3)
    pruned = result['steps'].get('pruned', {})
    statements = result['steps'].get('statements', [])
    
    with col1:
        st.metric(
            "Pruning Rate", 
            f"{pruned.get('pruned_count', 0)/pruned.get('original_count', 1)*100:.1f}%",
            help="Percentage of utterances kept (clinically relevant)"
        )
    with col2:
        st.metric(
            "Statements Generated", 
            len(statements),
            help="Number of Q&A pairs converted to statements"
        )
    with col3:
        signals = result['steps'].get('phq9_signals', {})
        st.metric(
            "PHQ-9 Signals", 
            len(signals),
            help="Number of PHQ-9 categories detected"
        )
    
    # Show WSS and other scores
    if statements:
        wss_scores = result['steps'].get('wss_scores', {})
        if wss_scores:
            wss_df = pd.DataFrame({
                'Statement': list(wss_scores.keys())[:5],
                'WSS Score': [f"{v:.4f}" for v in list(wss_scores.values())[:5]]
            })
            st.dataframe(wss_df, use_container_width=True)

# ===================== TAB 3: Evaluation =====================
with tab3:
    st.markdown('<p class="sub-header">📈 Evaluation Metrics</p>', unsafe_allow_html=True)
    
    # Run evaluation
    with st.spinner("📊 Computing evaluation metrics..."):
        summarizer = KiASSummarizer(window_size=window_size)
        result = summarizer.summarize(patient_data['transcript'], max_sentences=max_sentences)
        evaluator = Evaluator()
        
        summary_texts = result['summary']
        statements = result['steps'].get('statements', [])
        reference_texts = statements if statements else [s for _, s in patient_data['transcript']]
        
        # Compute all metrics
        jsd = evaluator.compute_jensen_shannon_divergence(summary_texts, reference_texts)
        sim = evaluator.compute_contextual_similarity(summary_texts, reference_texts)
        overlap = evaluator.compute_thematic_overlap(summary_texts, reference_texts)
        rouge2 = evaluator.compute_rouge(summary_texts, reference_texts, n=2)
    
    # Display metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Key Metrics")
        metrics_data = {
            "Metric": ["Jensen-Shannon Divergence", "Contextual Similarity", 
                      "Thematic Overlap", "ROUGE-2 Precision", "ROUGE-2 Recall", "ROUGE-2 F1"],
            "Score": [f"{jsd:.4f}", f"{sim:.4f}", f"{overlap:.2%}", 
                     f"{rouge2['precision']:.4f}", f"{rouge2['recall']:.4f}", f"{rouge2['f1']:.4f}"],
            "Optimal": ["Lower", "Higher", "Higher", "Higher", "Higher", "Higher"]
        }
        st.table(metrics_data)
    
    with col2:
        st.markdown("### 📈 Metric Visualization")
        fig = go.Figure(data=[
            go.Bar(
                x=['JSD', 'Similarity', 'Thematic Overlap'],
                y=[jsd, sim, overlap],
                text=[f"{jsd:.3f}", f"{sim:.3f}", f"{overlap:.3f}"],
                textposition='auto',
                marker_color=['#e74c3c', '#2ecc71', '#3498db']
            )
        ])
        fig.update_layout(
            title="Key Evaluation Metrics",
            yaxis_title="Score",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# ===================== TAB 4: Comparative Charts =====================
with tab4:
    st.markdown('<p class="sub-header">📉 Comparative Analysis</p>', unsafe_allow_html=True)
    
    # Process all patients for comparison
    with st.spinner("🔄 Processing all patients for comparison..."):
        all_results = []
        summarizer = KiASSummarizer(window_size=window_size)
        evaluator = Evaluator()
        
        progress_bar = st.progress(0)
        for idx, (pid, pdata) in enumerate(data.items()):
            result = summarizer.summarize(pdata['transcript'], max_sentences=max_sentences)
            statements = result['steps'].get('statements', [])
            summary_texts = result['summary']
            reference_texts = statements if statements else [s for _, s in pdata['transcript']]
            
            all_results.append({
                'Patient': pdata['patient_id'],
                'Diagnosis': pdata['metadata']['diagnosis'],
                'PHQ-8': pdata['metadata']['phq8_score'],
                'Pruning Rate': result['steps'].get('pruned', {}).get('pruned_count', 0) / 
                               result['steps'].get('pruned', {}).get('original_count', 1) * 100,
                'Statements': len(statements),
                'JSD': evaluator.compute_jensen_shannon_divergence(summary_texts, reference_texts),
                'Similarity': evaluator.compute_contextual_similarity(summary_texts, reference_texts),
                'Thematic Overlap': evaluator.compute_thematic_overlap(summary_texts, reference_texts),
                'ROUGE-F1': evaluator.compute_rouge(summary_texts, reference_texts, n=2)['f1']
            })
            progress_bar.progress((idx + 1) / len(data))
        
        progress_bar.empty()
    
    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Chart 1: PHQ-8 vs Metrics
    st.markdown("### 📊 PHQ-8 Score vs Evaluation Metrics")
    
    fig1 = make_subplots(rows=2, cols=2, subplot_titles=(
        "JSD vs PHQ-8", "Similarity vs PHQ-8",
        "Thematic Overlap vs PHQ-8", "ROUGE-F1 vs PHQ-8"
    ))
    
    fig1.add_trace(go.Scatter(x=df_results['PHQ-8'], y=df_results['JSD'], 
                              mode='markers+text', text=df_results['Patient'],
                              textposition='top center', name='JSD'),
                   row=1, col=1)
    fig1.add_trace(go.Scatter(x=df_results['PHQ-8'], y=df_results['Similarity'],
                              mode='markers+text', text=df_results['Patient'],
                              textposition='top center', name='Similarity'),
                   row=1, col=2)
    fig1.add_trace(go.Scatter(x=df_results['PHQ-8'], y=df_results['Thematic Overlap'],
                              mode='markers+text', text=df_results['Patient'],
                              textposition='top center', name='Thematic Overlap'),
                   row=2, col=1)
    fig1.add_trace(go.Scatter(x=df_results['PHQ-8'], y=df_results['ROUGE-F1'],
                              mode='markers+text', text=df_results['Patient'],
                              textposition='top center', name='ROUGE-F1'),
                   row=2, col=2)
    
    fig1.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Chart 2: Performance Comparison (Paper-like Table)
    st.markdown("### 📋 Performance Comparison (Paper Table 3 Equivalent)")
    
    # Group by diagnosis
    diag_groups = df_results.groupby('Diagnosis').agg({
        'PHQ-8': 'mean',
        'Pruning Rate': 'mean',
        'JSD': 'mean',
        'Similarity': 'mean',
        'Thematic Overlap': 'mean',
        'ROUGE-F1': 'mean'
    }).round(4)
    
    st.dataframe(diag_groups, use_container_width=True)
    
    # Chart 3: All Patients Comparison
    st.markdown("### 📊 All Patients Comparison")
    
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name='PHQ-8', x=df_results['Patient'], y=df_results['PHQ-8'],
                          marker_color='#3498db'))
    fig2.add_trace(go.Bar(name='Thematic Overlap %', x=df_results['Patient'], 
                          y=df_results['Thematic Overlap'] * 100,
                          marker_color='#2ecc71'))
    fig2.add_trace(go.Bar(name='Pruning Rate %', x=df_results['Patient'], 
                          y=df_results['Pruning Rate'],
                          marker_color='#e74c3c'))
    
    fig2.update_layout(
        title="Patient Comparison: PHQ-8, Thematic Overlap, and Pruning Rate",
        xaxis_title="Patient",
        yaxis_title="Percentage / Score",
        barmode='group',
        height=500
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Summary statistics (Paper-like)
    st.markdown("### 📊 Overall Performance Summary (Paper Table 2 Equivalent)")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg JSD", f"{df_results['JSD'].mean():.4f}", 
                  help="Lower is better")
    with col2:
        st.metric("Avg Similarity", f"{df_results['Similarity'].mean():.4f}",
                  help="Higher is better")
    with col3:
        st.metric("Avg Thematic Overlap", f"{df_results['Thematic Overlap'].mean():.2%}",
                  help="Higher is better")
    with col4:
        st.metric("Avg ROUGE-F1", f"{df_results['ROUGE-F1'].mean():.4f}",
                  help="Higher is better")

# ===================== TAB 5: PHQ-9 Analysis =====================
with tab5:
    st.markdown('<p class="sub-header">🔬 PHQ-9 Signal Analysis</p>', unsafe_allow_html=True)
    
    # Process all patients for PHQ-9 signals
    with st.spinner("🔍 Analyzing PHQ-9 signals across all patients..."):
        preprocessor = DataPreprocessor()
        signal_matrix = defaultdict(lambda: defaultdict(int))
        
        for pid, pdata in data.items():
            statements = preprocessor.convert_to_statements(
                preprocessor.prune_conversation(pdata['transcript'])
            )
            signals = preprocessor.extract_phq9_signals(statements)
            for category, stmts in signals.items():
                signal_matrix[pid][category] = len(stmts)
    
    # Convert to DataFrame
    signal_categories = ['S1_anhedonia', 'S2_depressed_mood', 'S3_sleep', 'S4_fatigue',
                        'S5_appetite', 'S6_worthlessness', 'S7_concentration', 
                        'S8_psychomotor', 'S9_suicidal']
    
    signal_names = ['Anhedonia', 'Depressed Mood', 'Sleep', 'Fatigue', 
                   'Appetite', 'Worthlessness', 'Concentration', 'Psychomotor', 'Suicidal']
    
    # Create heatmap data
    heatmap_data = []
    patients = list(data.keys())
    for pid in patients:
        row = []
        for cat in signal_categories:
            row.append(signal_matrix[pid].get(cat, 0))
        heatmap_data.append(row)
    
    # Display heatmap
    st.markdown("### 🗺️ PHQ-9 Signal Detection Heatmap")
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=signal_names,
        y=[data[pid]['patient_id'] for pid in patients],
        colorscale='YlOrRd',
        text=heatmap_data,
        texttemplate='%{text}',
        textfont={"size": 10},
    ))
    
    fig.update_layout(
        title="PHQ-9 Signal Frequency by Patient",
        xaxis_title="PHQ-9 Categories",
        yaxis_title="Patient",
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Signal distribution chart
    st.markdown("### 📊 Signal Distribution Across Patients")
    
    # Aggregate signals
    total_signals = defaultdict(int)
    for pid in patients:
        for cat in signal_categories:
            total_signals[cat] += signal_matrix[pid].get(cat, 0)
    
    fig2 = go.Figure(data=[
        go.Bar(
            x=signal_names,
            y=[total_signals[cat] for cat in signal_categories],
            marker_color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', 
                         '#9b59b6', '#1abc9c', '#e67e22', '#e74c3c', '#8e44ad']
        )
    ])
    
    fig2.update_layout(
        title="Total PHQ-9 Signal Mentions Across All Patients",
        xaxis_title="PHQ-9 Categories",
        yaxis_title="Total Mentions",
        height=400
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # PHQ-9 details per patient
    st.markdown("### 📋 PHQ-9 Signal Details by Patient")
    
    signal_data = []
    for pid in patients:
        row = {'Patient': data[pid]['patient_id'], 'Diagnosis': data[pid]['metadata']['diagnosis']}
        for cat, name in zip(signal_categories, signal_names):
            row[name] = signal_matrix[pid].get(cat, 0)
        signal_data.append(row)
    
    signal_df = pd.DataFrame(signal_data)
    st.dataframe(signal_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
**📖 Reference:** Manas G, Aribandi V, Kursuncu U, et al. 
Knowledge-Infused Abstractive Summarization of Clinical Diagnostic Interviews: 
Framework Development Study. JMIR Ment Health 2021;8(5):e20865
""")
