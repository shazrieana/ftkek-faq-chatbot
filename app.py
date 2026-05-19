# ==========================================
# 1. IMPORT LIBRARIES & INITIAL CONFIG
# ==========================================
import streamlit as st  # Web framework
import pandas as pd     # Data handling
from sentence_transformers import SentenceTransformer, util  # AI model

# FORCE MAXIMUM SCREEN WIDTH & COLLAPSE SIDEBAR (Dashboard Feel)
st.set_page_config(
    page_title="FTKEK Student Portal AI Assistant", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. ADVANCED CUSTOM CSS (Visual Theme matching your screenshot)
# ==========================================
st.markdown("""
    <style>
        /* Main background color adjustment */
        .stApp { background-color: #f7f9fc; }

        /* Remove default Streamlit top padding for a clean 'navbar' look */
        .block-container {
            padding-top: 2rem; 
            padding-bottom: 0rem; 
            padding-left: 2rem; 
            padding-right: 2rem;
        }

        /* Styling for titles and subheaders */
        h1 { color: #004A99; font-weight: 700; margin-bottom: 0.5rem; }
        h3 { color: #004A99; margin-top: 1rem; margin-bottom: 0.5rem; }

        /* --- PORTAL CARD SIMULATION STYLING --- */
        .portal-card-container {
            border: 1px solid #e1e8ed;
            border-radius: 12px;
            background-color: white;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        }
        .portal-card-container:hover {
            box-shadow: 0 8px 15px rgba(0,0,0,0.1);
            transform: translateY(-3px);
        }
        .portal-card-icon {
            font-size: 2.5rem;
            margin-right: 1rem;
            display: inline-block;
            vertical-align: middle;
        }
        .portal-card-text {
            display: inline-block;
            vertical-align: middle;
            max-width: calc(100% - 4rem);
        }
        .portal-card-title {
            color: #004A99;
            font-weight: 600;
            font-size: 1.25rem;
            margin-bottom: 0.25rem;
        }
        .portal-card-desc { color: #6c757d; font-size: 0.95rem; }

        /* --- CHATBOT BUBBLE STYLING --- */
        .stChatMessage { border-radius: 15px; margin-bottom: 10px; padding: 10px; }
        /* User bubbles */
        .stChatMessage.user { background-color: #e6f7ff; border: 1px solid #b3e6ff; }
        /* Assistant bubbles */
        .stChatMessage.assistant { background-color: white; border: 1px solid #e1e8ed; }

        /* Fix positioning and style the chat input bar */
        .stChatInputContainer {
            background-color: transparent !important;
            bottom: 30px !important;
            padding-left: 0 !important; padding-right: 0 !important;
        }
        .stChatInput {
            border-radius: 25px !important;
            border: 1px solid #ccd0d5 !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LOAD BRAIN & DATA (CACHED)
# ==========================================
@st.cache_resource
def load_chatbot():
    # Attempt to read data, handling potential deployment pathing issues
    try:
        df = pd.read_csv("knowledge_base.csv")
    except FileNotFoundError:
        # Fallback for complex paths if needed, but basic upload is usually fine
        df = pd.read_csv("/mount/src/Chatbot_FTKEK/knowledge_base.csv")
    
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Load SBERT model
    corpus_embeddings = model.encode(df['Question'].tolist(), convert_to_tensor=True)  # Encode FAQs
    return df, model, corpus_embeddings

df, model, corpus_embeddings = load_chatbot()  # Initialize the tools

# ==========================================
# 4. DASHBOARD LAYOUT & PORTAL HUB (The Visual "Ugly" Fix)
# ==========================================
st.title("🌐 FTKEK Student Portal Terminal")
st.markdown("Welcome to the integrated FTKEK Academic Assistant and Resources Hub.")
st.divider()

# Create standard 60/40 Split
col_portal, col_chat = st.columns([6, 4], gap="large")

# --- LEFT VIEWPORT: Simulated Portal Interface ---
with col_portal:
    st.markdown("### 🏢 Core Faculty Resource Modules")
    st.write("Use the interactive links below to browse official data streams.")

    # Definition of a reusable card component function
    def portal_card(icon, title, desc, url):
        st.markdown(f"""
            <a href="{url}" target="_blank" style="text-decoration: none; color: inherit;">
                <div class="portal-card-container">
                    <div class="portal-card-icon">{icon}</div>
                    <div class="portal-card-text">
                        <div class="portal-card-title">{title}</div>
                        <div class="portal-card-desc">{desc}</div>
                    </div>
                </div>
            </a>
        """, unsafe_allow_html=True)

    # Rendering the cards that look like standard website navigation blocks
    portal_card(
        "🎓", "Academic Mobility Portal", 
        "Explore international exchange programs, criteria, and scholarship application details.",
        "https://blog.utem.edu.my/ftkekstudentmobility/"
    )
    portal_card(
        "📜", "Academic Handbook", 
        "Official UTeM reference for degree guidelines, course registration, and assessment rules.",
        "https://www.utem.edu.my/" # Replace with direct handbook link if you have it
    )
    portal_card(
        "🔔", "FTKEK Notice Board", 
        "Live announcements, departmental updates, and student activity notices.",
        "https://ftkek.utem.edu.my/"
    )

# ==========================================
# 5. CHATBOT ASSISTANT viewport (Indented for col_chat)
# ==========================================
with col_chat:
    st.markdown("### 💬 SBERT Conversational Assistant")
    st.write("Ask academic questions or use our persistent memory menus.")

    # CHAT HISTORY & STATE INITIALIZATION
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm FTKEK AI Assistant. Your guide for academics and faculty services. What can I help you with? 😊"}
        ]

    # Initialize state flags
    if "waiting_for_programme_type" not in st.session_state: st.session_state.waiting_for_programme_type = False
    if "waiting_for_dean_type" not in st.session_state: st.session_state.waiting_for_dean_type = False
    if "waiting_for_head_type" not in st.session_state: st.session_state.waiting_for_head_type = False    

    # DISPLAY MESSAGES (Bubble theme defined in CSS)
    # Important optimization: Use key when iterating
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"], key=f"msg_{idx}")

    # CHAT INPUT
    if prompt := st.chat_input("Type your message..."):
        # Save user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        user_input = prompt.lower().strip()

        # [REMAINDER OF YOUR CHATBOT STATE LOGIC - Kept identical, but now indented here]
        # (Branching for Dean, Program, etc.)

        # --- STATE BRANCH 1: PROGRAMME Menu ---
        if st.session_state.waiting_for_programme_type:
            if "degree" in user_input:
                ans = df[df['Question'] == 'what bachelor degree programmes are offered by ftkek?']['Answer'].values[0]
                response = f"{ans}\n\n*You are still in the Programmes topic. You can check another one (Diploma, Master, PhD) or ask a new question.*"
            elif "diploma" in user_input:
                ans = df[df['Question'] == 'what diploma programmes are available at ftkek?']['Answer'].values[0]
                response = f"{ans}\n\n*You are still in the Programmes topic. You can check another one (Degree, Master, PhD) or ask a new question.*"
            elif "master" in user_input or "master's degree" in user_input:
                ans = df[df['Question'] == 'what master degree programmes are available at ftkek?']['Answer'].values[0]
                response = f"{ans}\n\n*You are still in the Programmes topic. You can check another one (Diploma, Degree, PhD) or ask a new question.*"
            elif "phd" in user_input or "doctor" in user_input:
                ans = df[df['Question'] == 'what phd programmes does ftkek offer?']['Answer'].values[0]
                response = f"{ans}\n\n*You are still in the Programmes topic. You can check another one (Diploma, Degree, Master) or ask a new question.*"
            else:
                st.session_state.waiting_for_programme_type = False
                query_embedding = model.encode(prompt, convert_to_tensor=True)
                hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=1)
                best_match_index = hits[0][0]['corpus_id']
                score = hits[0][0]['score']
                if score >= 0.60: response = df.iloc[best_match_index]['Answer']
                else: response = "Sorry, I don't have that answer. Refer to the FTKEK office at +606-270 2112."

        # --- STATE BRANCH 2: DEAN Menu ---
        elif st.session_state.waiting_for_dean_type:
            if "main" in user_input or "faculty leader" in user_input:
                ans = df[df['Question'] == 'who is the dean of FTKEK?']['Answer'].values[0]
                response = f"{ans}\n\n*You are still in the Dean topic. Look up Deputies or ask something else.*"
            elif "academic" in user_input:
                ans = df[df['Question'] == 'who is the deputy dean of academic affairs?']['Answer'].values[0]
                response = f"{ans}\n\n*You are still in the Dean topic. Look up Deans (Main Dean, Research, Student Affairs).* "
            elif "research" in user_input:
                ans = df[df['Question'] == 'who is the deputy dean of research adn postgraduates studies?']['Answer'].values[0]
                response = f"{ans}\n\n*You are still in the Dean topic.*"
            elif "student" in user_input:
                ans = df[df['Question'] == 'who is the deputy dean of student and alumni?']['Answer'].values[0]
                response = f"{ans}\n\n*You are still in the Dean topic.*"
            else:
                st.session_state.waiting_for_dean_type = False
                query_embedding = model.encode(prompt, convert_to_tensor=True)
                hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=1)
                best_match_index = hits[0][0]['corpus_id']
                score = hits[0][0]['score']
                if score >= 0.60: response = df.iloc[best_match_index]['Answer']
                else: response = "Sorry, I don't have that answer."

        # --- STATE BRANCH 3: HOD Menu ---
        elif st.session_state.waiting_for_head_type:
            if "engineering technology" in user_input:
                ans = df[df['Question'] == 'who is the deparment head of engineering technology?']['Answer'].values[0]
                response = f"{ans}\n\n*Still looking at HoDs.*"
            elif "engineering" in user_input:
                ans = df[df['Question'] == 'who is the deparment head of engineering?']['Answer'].values[0]
                response = f"{ans}\n\n*Still looking at HoDs.*"
            elif "technology" in user_input:
                ans = df[df['Question'] == 'who is the deparment head of technology?']['Answer'].values[0]
                response = f"{ans}\n\n*Still looking at HoDs.*"
            elif "diploma" in user_input:
                ans = df[df['Question'] == 'who is the deparment head of diploma?']['Answer'].values[0]
                response = f"{ans}\n\n*Still looking at HoDs.*"
            else:
                st.session_state.waiting_for_head_type = False
                query_embedding = model.encode(prompt, convert_to_tensor=True)
                hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=1)
                if hits[0][0]['score'] >= 0.60: response = df.iloc[hits[0][0]['corpus_id']]['Answer']
                else: response = "Sorry, I don't have that answer."

        # --- STATE BRANCH 4: Detective Keywords ---
        elif "programme" in user_input:
            st.session_state.waiting_for_dean_type = False; st.session_state.waiting_for_head_type = False
            response = df[df['Question'] == 'what programmes does ftkek offer?']['Answer'].values[0]
            st.session_state.waiting_for_programme_type = True

        elif "dean" in user_input:
            st.session_state.waiting_for_programme_type = False; st.session_state.waiting_for_head_type = False
            response = "Deans available: Main Dean, Academic, Research, Student Affairs. Which one?"
            st.session_state.waiting_for_dean_type = True

        elif "head" in user_input and "department" in user_input:
            st.session_state.waiting_for_programme_type = False; st.session_state.waiting_for_dean_type = False
            response = "Departments: Engineering, Engineering Technology, Technology, Diploma. Which one?"
            st.session_state.waiting_for_head_type = True

        # --- STATE BRANCH 5: SBERT Vector Search ---
        else:
            st.session_state.waiting_for_programme_type = False; st.session_state.waiting_for_dean_type = False; st.session_state.waiting_for_head_type = False
            query_embedding = model.encode(prompt, convert_to_tensor=True)
            hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=1)
            best_match_index = hits[0][0]['corpus_id']
            score = hits[0][0]['score']

            if score >= 0.60:
                response = df.iloc[best_match_index]['Answer']
                if best_match_index == 36: st.session_state.waiting_for_programme_type = True
                elif best_match_index == 46: st.session_state.waiting_for_head_type = True
            else:
                response = "Sorry, I don't have that answer."

        # Show response and save
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})