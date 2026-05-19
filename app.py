# ==========================================
# 1. IMPORT LIBRARIES & INITIAL CONFIG
# ==========================================
import streamlit as st  # Web framework
import pandas as pd     # Data handling
from sentence_transformers import SentenceTransformer, util  # AI model

# FORCE MAXIMUM SCREEN WIDTH (Must be the absolute first Streamlit command)
st.set_page_config(
    page_title="FTKEK Student Portal AI", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# Custom minimal CSS styling to match layout bubbles and fix the viewport scroll padding
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem; 
            padding-bottom: 80px !important; 
            padding-left: 2rem; 
            padding-right: 2rem;
        }
        h3 {margin-bottom: 0.5rem;}
        
        /* Your chat bubble styles */
        .stChatMessage { 
            background-color: #f0f2f6; 
            border-radius: 15px; 
            padding: 10px; 
            margin-bottom: 10px; 
        }
        
        /* Push the typing container up from the bottom edge slightly */
        .stChatInputContainer { 
            bottom: 20px !important;  
            padding-left: 10px !important;
            padding-right: 10px !important;
        }
    </style>
""", unsafe_html=True)

# ==========================================
# 2. LOAD BRAIN & DATA (CACHED)
# ==========================================
@st.cache_resource
def load_chatbot():
    df = pd.read_csv("knowledge_base.csv")  # Read your CSV
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Load SBERT
    corpus_embeddings = model.encode(df['Question'].tolist(), convert_to_tensor=True)  # Encode FAQs
    return df, model, corpus_embeddings

df, model, corpus_embeddings = load_chatbot()  # Initialize tools

# ==========================================
# 3. CREATE SIDE-BY-SIDE INTERFACE SPLIT
# ==========================================
# 60% Width for Website Viewport, 40% Width for Chatbot Assistant Viewport
col_web, col_chat = st.columns([6, 4])

# --- LEFT VIEWPORT: The Official Faculty Web Portal ---
with col_web:
    st.markdown("### 🌐 FTKEK Official Web Portal")
    st.components.v1.iframe(
        "https://blog.utem.edu.my/ftkekstudentmobility/", 
        height=720, 
        scrolling=True
    )

# --- RIGHT VIEWPORT: Your Closed-Domain AI Assistant ---
with col_chat:
    st.markdown("### 💬 FTKEK AI Assistant")
    
    # 4. CHAT HISTORY & STATE INITIALIZATION
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm FTKEK AI Assistant. Your guide for academics and faculty services. What can I help you with? 😊"}
        ]

    # Initialize the state machine flags
    if "waiting_for_programme_type" not in st.session_state:
        st.session_state.waiting_for_programme_type = False

    if "waiting_for_dean_type" not in st.session_state:
        st.session_state.waiting_for_dean_type = False

    if "waiting_for_head_type" not in st.session_state:
        st.session_state.waiting_for_head_type = False    

    # 5. DISPLAY MESSAGES (The "Bubble" look inside column 2)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 6. CHAT INPUT & STATE RETRIEVAL LOGIC
    if prompt := st.chat_input("Type your message..."):
        # Show user's message immediately on screen
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Convert the prompt to lowercase and clean spaces for comparison
        user_input = prompt.lower().strip()

        # --- STATE BRANCH 1: Handle the PERSISTENT PROGRAMME menu choice if active ---
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
                if score >= 0.60:
                    response = df.iloc[best_match_index]['Answer']
                else:
                    response = "Sorry, I don't have that answer, but you can refer to the FTKEK admin office at +606-270 2112 or email ftkek@utem.edu.my."

        # --- STATE BRANCH 2: Handle the PERSISTENT DEAN menu choice if active ---
        elif st.session_state.waiting_for_dean_type:
            if "main" in user_input or "faculty leader" in user_input:
                ans = df[df['Question'] == 'who is the dean of FTKEK?']['Answer'].values[0]
                response = f"{ans}\n\n*You are still in the Dean topic. You can look up other Deputy Deans (Academic, Research, Student Affairs) or ask a new question.*"
            elif "academic" in user_input:
                ans = df[df['Question'] == 'who is the deputy dean of academic affairs?']['Answer'].values[0]
                response = f"{ans}\n\n*You are still in the Dean topic. You can look up other Deans (Main Dean, Research, Student Affairs) or ask a new question.*"
            elif "research" in user_input or "postgraduate" in user_input:
                ans = df[df['Question'] == 'who is the deputy dean of research adn postgraduates studies?']['Answer'].values[0]
                response = f"{ans}\n\n*You are still in the Dean topic. You can look up other Deans (Main Dean, Academic, Student Affairs) or ask a new question.*"
            elif "student" in user_input or "alumni" in user_input:
                ans = df[df['Question'] == 'who is the deputy dean of student and alumni?']['Answer'].values[0]
                response = f"{ans}\n\n*You are still in the Dean topic. You can look up other Deans (Main Dean, Academic, Research) or ask a new question.*"
            else:
                st.session_state.waiting_for_dean_type = False
                
                query_embedding = model.encode(prompt, convert_to_tensor=True)
                hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=1)
                best_match_index = hits[0][0]['corpus_id']
                score = hits[0][0]['score']
                if score >= 0.60:
                    response = df.iloc[best_match_index]['Answer']
                else:
                    response = "Sorry, I don't have that answer, but you can refer to the FTKEK admin office at +606-270 2112 or email ftkek@utem.edu.my."

        # --- STATE BRANCH 3: Handle the PERSISTENT HOD menu choice if active ---
        elif st.session_state.waiting_for_head_type:
            if "engineering" in user_input:
                if "technology" in user_input:
                    ans = df[df['Question'] == 'who is the deparment head of engineering technology?']['Answer'].values[0]
                    response = f"{ans}\n\n*You are still looking at Head of Departments. You can choose another one (Engineering, Technology, Diploma) or ask something else.*"
                else:
                    ans = df[df['Question'] == 'who is the deparment head of engineering?']['Answer'].values[0]
                    response = f"{ans}\n\n*You are still looking at Head of Departments. You can choose another one (Engineering Technology, Technology, Diploma) or ask something else.*"
            elif "technology" in user_input:
                ans = df[df['Question'] == 'who is the deparment head of technology?']['Answer'].values[0]
                response = f"{ans}\n\n*You are still looking at Head of Departments. You can choose another one (Engineering, Engineering Technology, Diploma) or ask something else.*"
            elif "diploma" in user_input:
                ans = df[df['Question'] == 'who is the deparment head of diploma?']['Answer'].values[0]
                response = f"{ans}\n\n*You are still looking at Head of Departments. You can choose another one (Engineering, Engineering Technology, Technology) or ask something else.*"
            else:
                st.session_state.waiting_for_head_type = False
                
                query_embedding = model.encode(prompt, convert_to_tensor=True)
                hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=1)
                best_match_index = hits[0][0]['corpus_id']
                score = hits[0][0]['score']
                if score >= 0.60:
                    response = df.iloc[best_match_index]['Answer']
                else:
                    response = "Sorry, I don't have that answer, but you can refer to the FTKEK admin office at +606-270 2112 or email ftkek@utem.edu.my."

        # --- STATE BRANCH 4: Detect initial keyword triggers ---
        elif "programme" in user_input:
            st.session_state.waiting_for_dean_type = False
            st.session_state.waiting_for_head_type = False
            response = df[df['Question'] == 'what programmes does ftkek offer?']['Answer'].values[0]
            st.session_state.waiting_for_programme_type = True

        elif "dean" in user_input:
            st.session_state.waiting_for_programme_type = False
            st.session_state.waiting_for_head_type = False
            response = "We have a **Main Dean** and three **Deputy Deans**. Which one are you looking for? (Main Dean, Academic, Research, or Student Affairs)"
            st.session_state.waiting_for_dean_type = True

        elif "head" in user_input and "department" in user_input:
            st.session_state.waiting_for_programme_type = False
            st.session_state.waiting_for_dean_type = False
            response = "We have **Engineering**, **Engineering Technology**, **Technology**, and **Diploma** Departments. Which one are you looking for?"
            st.session_state.waiting_for_head_type = True

        # --- STATE BRANCH 5: SBERT search with Confidence Threshold ---
        else:
            st.session_state.waiting_for_programme_type = False
            st.session_state.waiting_for_dean_type = False
            st.session_state.waiting_for_head_type = False
            
            query_embedding = model.encode(prompt, convert_to_tensor=True)
            hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=1)
            
            best_match_index = hits[0][0]['corpus_id']
            score = hits[0][0]['score']

            if score >= 0.60:
                response = df.iloc[best_match_index]['Answer']
                if best_match_index == 36:
                    st.session_state.waiting_for_programme_type = True
                elif best_match_index == 46:
                    st.session_state.waiting_for_head_type = True
            else:
                response = "Sorry, I don't have that answer, but you can refer to the FTKEK admin office at +606-270 2112 or email ftkek@utem.edu.my."

        # Show AI response on screen and save it to memory history
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})