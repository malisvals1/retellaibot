import streamlit as st
import pandas as pd
import requests
import re  # For phone validation
import time  # For delays in batch calling
import threading  # For non-blocking batch operations
import json  # For config files
from dotenv import load_dotenv  # For secure API key handling
import os

# Load environment variables if available
load_dotenv()

# Streamlit app title
st.title("Sales Bot App - Enhanced for Marketing & Scaling")

# Initialize session state
if 'retell_api_key' not in st.session_state:
    st.session_state.retell_api_key = os.getenv("RETELL_API_KEY", "")
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")
if 'from_number' not in st.session_state:
    st.session_state.from_number = os.getenv("FROM_NUMBER", "")
if 'llm_id' not in st.session_state:
    st.session_state.llm_id = None
if 'agent_id' not in st.session_state:
    st.session_state.agent_id = None
if 'leads' not in st.session_state:
    st.session_state.leads = []
if 'call_logs' not in st.session_state:
    st.session_state.call_logs = []
if 'product_description' not in st.session_state:
    st.session_state.product_description = ""
if 'custom_prompt' not in st.session_state:
    st.session_state.custom_prompt = ""
if 'batch_delay' not in st.session_state:
    st.session_state.batch_delay = 10
if 'status' not in st.session_state:
    st.session_state.status = ""

# Input fields
st.subheader("API Keys and Config")
st.session_state.retell_api_key = st.text_input("Retell AI API Key", value=st.session_state.retell_api_key)
st.session_state.openai_api_key = st.text_input("OpenAI API Key", value=st.session_state.openai_api_key, type="password")
st.session_state.from_number = st.text_input("From Number (E.164 format)", value=st.session_state.from_number)

# Product description
st.subheader("Product/Service Description")
st.session_state.product_description = st.text_area("Product/Service Description (for pitch generation)", value=st.session_state.product_description, height=100)

# Custom prompt
st.subheader("Custom Pitch Generation Prompt")
st.session_state.custom_prompt = st.text_area("Custom Pitch Generation Prompt (optional; use {name}, {info}, {product} as placeholders)", value=st.session_state.custom_prompt, height=150)

# Batch delay
st.session_state.batch_delay = st.number_input("Batch Call Delay (seconds)", value=st.session_state.batch_delay, min_value=1)

# Setup Agent
if st.button("Setup Agent"):
    if not all([st.session_state.retell_api_key, st.session_state.openai_api_key, st.session_state.from_number]):
        st.error("All fields are required.")
    elif not re.match(r'^\+\d{1,15}$', st.session_state.from_number):
        st.error("From number must be in E.164 format, e.g., +1234567890")
    else:
        try:
            # Create LLM
            url = "https://api.retellai.com/create-retell-llm"
            headers = {"Authorization": f"Bearer {st.session_state.retell_api_key}", "Content-Type": "application/json"}
            body = {
                "model": "gpt-4o",
                "general_prompt": "You are a sales agent calling {{name}}. Your goal is to sell using the following pitch: {{pitch}}. Be engaging, ask questions, handle objections, and try to close the sale."
            }
            resp = requests.post(url, headers=headers, json=body)
            resp.raise_for_status()
            st.session_state.llm_id = resp.json()['llm_id']
            
            # Create Agent
            url = "https://api.retellai.com/create-agent"
            body = {
                "response_engine": {"type": "retell-llm", "llm_id": st.session_state.llm_id},
                "agent_name": "SalesBot",
                "voice_id": "openai-Alloy",
                "language": "en-US"
            }
            resp = requests.post(url, headers=headers, json=body)
            resp.raise_for_status()
            st.session_state.agent_id = resp.json()['agent_id']
            
            # Bind Agent to Phone
            url = f"https://api.retellai.com/update-phone-number/{st.session_state.from_number}"
            body = {
                "outbound_agent_id": st.session_state.agent_id
            }
            resp = requests.patch(url, headers=headers, json=body)
            resp.raise_for_status()
            
            st.success("Agent setup complete.")
            st.session_state.status = "Agent ready."
        except Exception as e:
            st.error(f"Setup failed: {str(e)}")

st.text(st.session_state.status)

# Load CSV
st.subheader("Load Leads CSV")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None and st.button("Load Uploaded CSV"):
    try:
        df = pd.read_csv(uploaded_file)
        if 'phone' not in df.columns:
            st.error("CSV must have at least 'phone' column.")
        else:
            # Dedupe and validate phones
            df = df.drop_duplicates(subset=['phone'])
            df['phone'] = df['phone'].apply(lambda x: str(x).strip())
            # Automatically add +1 if the phone doesn't start with + and is digits (assuming US numbers)
            df['phone'] = df['phone'].apply(lambda x: f"+1{x}" if not x.startswith('+') and x.isdigit() else x)
            valid_phones = df['phone'].apply(lambda x: bool(re.match(r'^\+\d{1,15}$', x)))
            if not valid_phones.all():
                invalid_indices = df[~valid_phones].index
                for idx in invalid_indices:
                    original_phone = df.at[idx, 'phone']
                    fixed = ai_reformat_phone(original_phone)
                    if fixed and re.match(r'^\+\d{1,15}$', fixed):
                        df.at[idx, 'phone'] = fixed
                # Re-validate
                valid_phones = df['phone'].apply(lambda x: bool(re.match(r'^\+\d{1,15}$', x)))
                invalid = df[~valid_phones]['phone'].tolist()
                if invalid:
                    st.warning(f"Some phones could not be fixed and are skipped: {', '.join(invalid)}")
                df = df[valid_phones]
            st.session_state.leads = df.to_dict(orient='records')
            st.success(f"Loaded {len(st.session_state.leads)} leads.")
    except Exception as e:
        st.error(f"Failed to load CSV: {str(e)}")

# Display Leads
if st.session_state.leads:
    st.subheader("Leads")
    lead_df = pd.DataFrame(st.session_state.leads)
    st.data_frame(lead_df)

# Generate Pitch
st.subheader("Generate Pitch")
selected_lead_index = st.selectbox("Select Lead", options=range(len(st.session_state.leads)), format_func=lambda i: f"{st.session_state.leads[i].get('name', 'Unknown')} - {st.session_state.leads[i]['phone']}")
if st.button("Generate Pitch for Selected"):
    if 'leads' in st.session_state and st.session_state.leads:
        lead = st.session_state.leads[selected_lead_index]
        name = lead.get('name', 'the customer')
        info = lead.get('info', 'no specific information available')
        product = st.session_state.product_description or "our amazing product/service"
        
        custom_prompt = st.session_state.custom_prompt
        if custom_prompt:
            try:
                user_content = custom_prompt.format(name=name, info=info, product=product)
            except KeyError:
                st.error("Custom prompt should use {name}, {info}, {product} if needed.")
        else:
            user_content = f"Come up with the very best thing to sell to {name} based on {info}. The product is {product}. Generate a compelling pitch script for a phone call."
        
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {st.session_state.openai_api_key}", "Content-Type": "application/json"}
        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a creative sales pitch generator."},
                {"role": "user", "content": user_content}
            ]
        }
        try:
            resp = requests.post(url, headers=headers, json=body)
            resp.raise_for_status()
            pitch = resp.json()['choices'][0]['message']['content']
            lead['pitch'] = pitch
            st.text_area("Generated Pitch (Editable)", value=pitch, height=200, key="pitch_display")
            st.success("Pitch generated.")
        except Exception as e:
            st.error(f"Failed to generate pitch: {str(e)}")

# Initiate Call
if st.button("Confirm and Call Selected"):
    if 'leads' in st.session_state and st.session_state.leads:
        lead = st.session_state.leads[selected_lead_index]
        if 'pitch' not in lead:
            st.error("Generate pitch first.")
        else:
            try:
                url = "https://api.retellai.com/v2/create-phone-call"
                headers = {"Authorization": f"Bearer {st.session_state.retell_api_key}", "Content-Type": "application/json"}
                body = {
                    "from_number": st.session_state.from_number,
                    "to_number": lead['phone'],
                    "retell_llm_dynamic_variables": {
                        "name": lead.get('name', ''),
                        "pitch": lead['pitch']
                    }
                }
                resp = requests.post(url, headers=headers, json=body)
                resp.raise_for_status()
                call_id = resp.json().get('call_id', 'Unknown')
                st.success("Call initiated.")
                st.session_state.call_logs.append({
                    'phone': lead['phone'],
                    'name': lead.get('name', 'Unknown'),
                    'time': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'status': 'Initiated',
                    'call_id': call_id
                })
            except Exception as e:
                st.error(f"Failed to initiate call: {str(e)}")

# Display Call Logs
if st.session_state.call_logs:
    st.subheader("Call Logs")
    log_df = pd.DataFrame(st.session_state.call_logs)
    st.data_frame(log_df)

# Export Logs
if st.button("Export Logs to CSV"):
    if st.session_state.call_logs:
        df = pd.DataFrame(st.session_state.call_logs)
        csv = df.to_csv(index=False)
        st.download_button("Download Logs CSV", csv, "call_logs.csv", "text/csv")
    else:
        st.error("No logs to export.")

# AI Reformat Phone (for reference, used in load_csv)
def ai_reformat_phone(phone):
    openai_api_key = st.session_state.openai_api_key
    if not openai_api_key:
        return None
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {openai_api_key}", "Content-Type": "application/json"}
    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a phone number formatter. Output only the formatted E.164 phone number, nothing else."},
            {"role": "user", "content": f"Format this phone number to E.164 format, assuming US if no country code: {phone}"}
        ]
    }
    try:
        resp = requests.post(url, headers=headers, json=body)
        resp.raise_for_status()
        formatted = resp.json()['choices'][0]['message']['content'].strip()
        return formatted
    except Exception as e:
        st.error(f"AI format error: {str(e)}")
