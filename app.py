"""
Hospital AI Auto-Reply Agent
============================
A simple AI assistant for a hospital that responds to patient queries using 
Google's Gemma 2b model via Ollama (local LLM).

This application:
- Accepts patient messages through a Streamlit web interface
- Uses Gemma 2b model for generating helpful responses
- Stores patient details in a JSON file
- Maintains conversation memory for contextual responses
"""

import streamlit as st
import json
import os
from datetime import datetime
import requests

# Configuration
DATA_FILE = "patients.json"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "gemma:2b"

# Page configuration
st.set_page_config(
    page_title="Hospital AI Assistant",
    page_icon="🏥",
    layout="centered"
)

def save_patient(data):
    """Save patient details to JSON file."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            patients = json.load(f)
    else:
        patients = []

    patients.append(data)

    with open(DATA_FILE, "w") as f:
        json.dump(patients, f, indent=2)

def get_ai_reply(message, conversation_history=None):
    """
    Get AI response from Ollama with Gemma 2b model.
    
    Args:
        message: The patient's message
        conversation_history: List of previous messages for context
    
    Returns:
        AI-generated response string
    """
    # Build conversation context
    system_prompt = """You are a helpful hospital assistant named "Hospital Helper".

Guidelines for responses:
- Always be polite, professional, and empathetic
- Assist with appointment booking inquiries
- Provide information about doctor availability
- Share hospital timings and services
- Keep responses concise but informative
- If you don't have specific information, ask for details or suggest contacting the hospital directly

Hospital Services:
- General Medicine: 24/7 available
- Cardiology: 9 AM - 5 PM, Monday to Saturday
- Dermatology: 10 AM - 6 PM, Monday to Friday
- Orthopedics: 8 AM - 4 PM, Monday to Saturday
- Pediatrics: 24/7 available
- Emergency: 24/7 available

Remember to be helpful and guide patients appropriately."""

    # Build messages array with conversation history
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history if available (for context)
    if conversation_history:
        for msg in conversation_history[-5:]:  # Last 5 messages for context
            messages.append(msg)
    
    # Add current message
    messages.append({"role": "user", "content": message})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data["message"]["content"]
        else:
            return "Sorry, I couldn't generate a response at this time. Please try again."
    except requests.exceptions.ConnectionError:
        return "Error: Unable to connect to Ollama. Please ensure Ollama is running on localhost:11434."
    except requests.exceptions.Timeout:
        return "Error: Request timed out. Please try again."
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize session state for conversation memory
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'patient_name' not in st.session_state:
    st.session_state.patient_name = ""

# Main UI
st.title("🏥 Hospital AI Assistant")
st.write("Welcome! Chat with our virtual assistant for appointments and hospital information.")

# Display welcome message
if not st.session_state.conversation_history:
    st.info("💬 Start a conversation by entering your message below.")

# Sidebar for clearing conversation
with st.sidebar:
    st.header("Options")
    if st.button("Clear Conversation"):
        st.session_state.conversation_history = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### ℹ️ Hospital Timings")
    st.markdown("""
    - **General Medicine**: 24/7
    - **Cardiology**: 9 AM - 5 PM (Mon-Sat)
    - **Dermatology**: 10 AM - 6 PM (Mon-Fri)
    - **Orthopedics**: 8 AM - 4 PM (Mon-Sat)
    - **Pediatrics**: 24/7
    - **Emergency**: 24/7
    """)

# Patient Information Form
st.subheader("📋 Patient Information")

col1, col2 = st.columns(2)
with col1:
    name = st.text_input("Patient Name", value=st.session_state.patient_name, placeholder="Enter your name")
with col2:
    phone = st.text_input("Phone Number", placeholder="Enter your phone number")

department = st.selectbox(
    "Department",
    ["General Medicine", "Cardiology", "Dermatology", "Orthopedics", "Pediatrics", "Emergency"]
)

# Message Input
message = st.text_area("💬 Your Message", placeholder="Type your message here... (e.g., I want to book an appointment)")

# Send Button
if st.button("📤 Send Message", type="primary"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Add user message to conversation history
        st.session_state.conversation_history.append({"role": "user", "content": message})
        
        # Get AI reply with conversation context
        reply = get_ai_reply(message, st.session_state.conversation_history)
        
        # Add AI response to conversation history
        st.session_state.conversation_history.append({"role": "assistant", "content": reply})
        
        # Save patient data
        patient_data = {
            "name": name if name else "Anonymous",
            "phone": phone if phone else "Not provided",
            "department": department,
            "message": message,
            "timestamp": str(datetime.now())
        }
        save_patient(patient_data)
        
        # Update patient name in session
        if name:
            st.session_state.patient_name = name
        
        # Display the response
        st.subheader("🤖 AI Assistant Reply")
        st.success(reply)
        
        st.markdown("---")
        st.markdown("*Thank you for contacting us! Please call 911 for emergencies.*")

# Display conversation history
if st.session_state.conversation_history:
    st.markdown("### 💭 Conversation History")
    for i, msg in enumerate(st.session_state.conversation_history):
        role = "You" if msg["role"] == "user" else "AI"
        st.markdown(f"**{role}:** {msg['content']}")

