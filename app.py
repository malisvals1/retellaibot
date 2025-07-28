import streamlit as st
import pandas as pd
import requests
import re  # For phone validation
import time  # For delays in batch calling
import threading  # For non-blocking batch operations
import json  # For config files
from dotenv import load_dotenv  # For secure API key handling
import os
import matplotlib.pyplot as plt  # For reports
from bs4 import BeautifulSoup  # For scraping
import multiprocessing  # For background scraping
import openai  # For OpenAI client
import random  # For A/B testing variants

# Load environment variables if available
load_dotenv()

# Persistent storage functions
def save_to_json(key, data):
    with open(f"{key}.json", "w") as f:
        json.dump(data, f)

def load_from_json(key, default=[]):
    filename = f"{key}.json"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return default

# Streamlit app title
st.title("Sales Bot App - Enhanced for Marketing & Scaling")

# Warning for compliance
st.warning("Ensure scraping complies with site terms and laws. Respect robots.txt and use ethically.")

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
    st.session_state.leads = load_from_json("leads")
if 'call_logs' not in st.session_state:
    st.session_state.call_logs = load_from_json("call_logs")
if 'product_description' not in st.session_state:
    st.session_state.product_description = ""
if 'custom_prompt' not in st.session_state:
    st.session_state.custom_prompt = ""
if 'batch_delay' not in st.session_state:
    st.session_state.batch_delay = 10
if 'status' not in st.session_state:
    st.session_state.status = ""
if 'niche' not in st.session_state:
    st.session_state.niche = ""
if 'scraping_instructions' not in st.session_state:
    st.session_state.scraping_instructions = ""
if 'closing_tips' not in st.session_state:
    st.session_state.closing_tips = ""
if 'background_scraping_process' not in st.session_state:
    st.session_state.background_scraping_process = None
if 'scrape_url' not in st.session_state:
    st.session_state.scrape_url = ""
if 'ai_output' not in st.session_state:
    st.session_state.ai_output = {}
if 'call_ids' not in st.session_state:
    st.session_state.call_ids = {}
if 'voice_id' not in st.session_state:
    st.session_state.voice_id = "openai-Alloy"

# Input fields
st.subheader("API Keys and Config")
st.session_state.retell_api_key = st.text_input("Retell AI API Key", value=st.session_state.retell_api_key)
st.session_state.openai_api_key = st.text_input("OpenAI API Key", value=st.session_state.openai_api_key, type="password")
st.session_state.from_number = st.text_input("From Number (E.164 format)", value=st.session_state.from_number)

# Voice selection
st.session_state.voice_id = st.selectbox("Voice", ["openai-Alloy", "11labs-Adrian", "openai-Echo", "openai-Fable"], index=0)

# Niche input
st.subheader("Niche")
st.session_state.niche = st.selectbox("Select Niche", options=["Real Estate", "SaaS", "Plumbing", "Other"], index=0)
if st.session_state.niche == "Other":
    st.session_state.niche = st.text_input("Custom Niche")

# Product description
st.subheader("Product/Service Description")
st.session_state.product_description = st.text_area("Product/Service Description (for pitch generation)", value=st.session_state.product_description, height=100)

# Custom prompt
st.subheader("Custom Pitch Generation Prompt")
st.session_state.custom_prompt = st.text_area("Custom Pitch Generation Prompt (optional; use {name}, {info}, {product} as placeholders)", value=st.session_state.custom_prompt, height=150)

# Batch delay
st.session_state.batch_delay = st.number_input("Batch Call Delay (seconds)", value=st.session_state.batch_delay, min_value=1)

# AI Orchestrator
st.subheader("AI Orchestrator")
if st.button("Run AI Orchestrator"):
    if st.session_state.openai_api_key and st.session_state.niche and st.session_state.product_description:
        try:
            client = openai.OpenAI(api_key=st.session_state.openai_api_key)
            prompt = f"You are a sales strategist. Analyze niche: {st.session_state.niche}, product: {st.session_state.product_description}. Output JSON: {{'scraping_instructions': 'str', 'pitch_script': 'str', 'closing_tips': 'str', 'websites_to_scrape': ['list of urls']}}"
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": prompt}],
                response_format={"type": "json_object"}
            )
            st.session_state.ai_output = json.loads(response.choices[0].message.content)
            st.session_state.scraping_instructions = st.session_state.ai_output.get('scraping_instructions', '')
            st.session_state.custom_prompt = st.session_state.ai_output.get('pitch_script', st.session_state.custom_prompt)
            st.session_state.closing_tips = st.session_state.ai_output.get('closing_tips', '')
            st.success("AI Orchestrator completed.")
            st.text_area("Scraping Instructions", value=st.session_state.scraping_instructions)
            st.text_area("Closing Tips", value=st.session_state.closing_tips)
        except Exception as e:
            st.error(f"AI Orchestrator failed: {str(e)}")
    else:
        st.error("Niche, product description, and OpenAI key required.")

# Setup Agent
if st.button("Setup Agent"):
    if not all([st.session_state.retell_api_key, st.session_state.openai_api_key, st.session_state.from_number]):
        st.error("All fields are required.")
    elif not re.match(r'^\+\d{1,15}$', st.session_state.from_number):
        st.error("From number must be in E.164 format, e.g., +1234567890")
    else:
        try:
            headers = {"Authorization": f"Bearer {st.session_state.retell_api_key}", "Content-Type": "application/json"}
            # Create LLM
            llm_body = {
                "model": "gpt-4o",
                "general_prompt": f"You are a sales agent calling {{name}}. Your goal is to sell using the following pitch: {{pitch}}. Be engaging, ask questions, handle objections, and try to close the sale. Closing tips: {st.session_state.closing_tips}"
            }
            llm_resp = requests.post("https://api.retellai.com/create-retell-llm", headers=headers, json=llm_body)
            llm_resp.raise_for_status()
            st.session_state.llm_id = llm_resp.json()['llm_id']
            
            # Create Agent
            agent_body = {
                "response_engine": {"type": "retell-llm", "llm_id": st.session_state.llm_id},
                "agent_name": "SalesBot",
                "voice_id": st.session_state.voice_id,
                "language": "en-US"
            }
            agent_resp = requests.post("https://api.retellai.com/create-agent", headers=headers, json=agent_body)
            agent_resp.raise_for_status()
            st.session_state.agent_id = agent_resp.json()['agent_id']
            
            # Bind Agent to Phone
            phone_body = {
                "outbound_agent_id": st.session_state.agent_id
            }
            phone_url = f"https://api.retellai.com/update-phone-number/{st.session_state.from_number}"
            phone_resp = requests.patch(phone_url, headers=headers, json=phone_body)
            phone_resp.raise_for_status()
            
            st.success("Agent setup complete.")
            st.session_state.status = "Agent ready."
        except Exception as e:
            st.error(f"Setup failed: {str(e)}")

st.text(st.session_state.status)

# Scrape Leads (enhanced with Yelp example)
def scrape_leads(url, niche):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        phones = []
        names = []
        for text in soup.find_all(text=True):
            phone_matches = re.findall(r'(\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})', text)
            if phone_matches:
                phones.extend(phone_matches)
            if re.match(r'^[A-Z][a-z]+\s[A-Z][a-z]+$', text.strip()):
                names.append(text.strip())
        new_leads = []
        phones = list(set(phones))
        for i, phone in enumerate(phones):
            fixed = ai_reformat_phone(phone)
            if fixed:
                name = names[i % len(names)] if names else "Unknown"
                new_leads.append({"phone": fixed, "name": name, "info": niche, "score": random.randint(1, 10)})
        return new_leads
    except Exception as e:
        return []

st.subheader("Scrape Leads")
st.session_state.scrape_url = st.text_input("Scrape URL (e.g., Yelp search)", value=st.session_state.scrape_url)
if st.button("Scrape Now"):
    if st.session_state.scrape_url:
        new_leads = scrape_leads(st.session_state.scrape_url, st.session_state.niche)
        st.session_state.leads.extend(new_leads)
        save_to_json("leads", st.session_state.leads)
        st.success(f"Added {len(new_leads)} leads.")

# Background Scraping
if st.checkbox("Enable Background Scraping"):
    def background_scraper(url, niche):
        while True:
            new_leads = scrape_leads(url, niche)
            st.session_state.leads.extend(new_leads)
            save_to_json("leads", st.session_state.leads)
            time.sleep(3600)  # 1 hour

    if st.session_state.background_scraping_process is None or not st.session_state.background_scraping_process.is_alive():
        st.session_state.background_scraping_process = multiprocessing.Process(target=background_scraper, args=(st.session_state.scrape_url, st.session_state.niche))
        st.session_state.background_scraping_process.start()
        st.success("Background scraping started.")
else:
    if st.session_state.background_scraping_process is not None and st.session_state.background_scraping_process.is_alive():
        st.session_state.background_scraping_process.terminate()
        st.session_state.background_scraping_process = None
        st.success("Background scraping stopped.")

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
            df['phone'] = df['phone'].apply(lambda x: f"+1{x}" if not x.startswith('+') and x.isdigit() else x)
            valid_phones = df['phone'].apply(lambda x: bool(re.match(r'^\+\d{1,15}$', x)))
            if not valid_phones.all():
                invalid_indices = df[~valid_phones].index
                for idx in invalid_indices:
                    original_phone = df.at[idx, 'phone']
                    fixed = ai_reformat_phone(original_phone)
                    if fixed and re.match(r'^\+\d{1,15}$', fixed):
                        df.at[idx, 'phone'] = fixed
                valid_phones = df['phone'].apply(lambda x: bool(re.match(r'^\+\d{1,15}$', x)))
                invalid = df[~valid_phones]['phone'].tolist()
                if invalid:
                    st.warning(f"Some phones could not be fixed and are skipped: {', '.join(invalid)}")
                df = df[valid_phones]
            st.session_state.leads.extend(df.to_dict(orient='records'))
            save_to_json("leads", st.session_state.leads)
            st.success(f"Loaded {len(df)} leads.")
    except Exception as e:
        st.error(f"Failed to load CSV: {str(e)}")

# Display Leads
if st.session_state.leads:
    st.subheader("Leads")
    lead_df = pd.DataFrame(st.session_state.leads)
    st.dataframe(lead_df)

# Lead Qualification & Enrichment
st.subheader("Qualify & Enrich Leads")
if st.button("Qualify All Leads"):
    if st.session_state.openai_api_key:
        client = openai.OpenAI(api_key=st.session_state.openai_api_key)
        for lead in st.session_state.leads:
            info = lead.get('info', '')
            prompt = f"Score this lead for niche {st.session_state.niche}: {info}. Score 1-10, and enrich with possible email or details."
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt}]
            )
            try:
                result = json.loads(response.choices[0].message.content)
                lead['score'] = result.get('score', 0)
                lead['enriched_info'] = result.get('enriched_info', info)
            except:
                lead['score'] = 0
                lead['enriched_info'] = "No enrichment"
        save_to_json("leads", st.session_state.leads)
        st.success("Leads qualified and enriched.")
    else:
        st.error("OpenAI key required.")

# Pitch Modes with A/B Testing
st.subheader("Pitch Generation")
pitch_mode = st.selectbox("Pitch Mode", ["Auto-Pitch", "Niche-Based", "Override", "A/B Testing"])
ab_pitch_a = ""
ab_pitch_b = ""
if pitch_mode == "A/B Testing":
    ab_pitch_a = st.text_area("Pitch A", value=st.session_state.custom_prompt)
    ab_pitch_b = st.text_area("Pitch B", "Alternative pitch version.")

selected_lead_index = st.selectbox("Select Lead for Pitch", options=range(len(st.session_state.leads)), format_func=lambda i: f"{st.session_state.leads[i].get('name', 'Unknown')} - {st.session_state.leads[i]['phone']}")
if st.button("Generate Pitch for Selected"):
    if st.session_state.leads:
        lead = st.session_state.leads[selected_lead_index]
        name = lead.get('name', 'the customer')
        info = lead.get('info', 'no specific information available')
        product = st.session_state.product_description or "our amazing product/service"
        
        custom_prompt = st.session_state.custom_prompt
        user_content = custom_prompt.format(name=name, info=info, product=product) if custom_prompt else f"Generate pitch for {name} based on {info}. Product: {product}."
        
        if pitch_mode == "Niche-Based":
            user_content += f" Tailor for niche: {st.session_state.niche}."
        elif pitch_mode == "A/B Testing":
            user_content += " Generate two variants: A and B for A/B testing."
        
        if st.session_state.openai_api_key:
            client = openai.OpenAI(api_key=st.session_state.openai_api_key)
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a creative sales pitch generator."},
                    {"role": "user", "content": user_content}
                ]
            )
            pitch = resp.choices[0].message.content
            lead['pitch'] = pitch
            st.text_area("Generated Pitch (Editable)", value=pitch, height=200, key="pitch_display")
            st.success("Pitch generated.")
        else:
            st.error("OpenAI key required.")

# Initiate Call (with A/B if enabled)
if st.button("Confirm and Call Selected"):
    if st.session_state.leads:
        lead = st.session_state.leads[selected_lead_index]
        if 'pitch' not in lead:
            st.error("Generate pitch first.")
        else:
            selected_pitch = lead['pitch']
            if pitch_mode == "A/B Testing" and ab_pitch_a and ab_pitch_b:
                selected_pitch = random.choice([ab_pitch_a, ab_pitch_b])
            try:
                url = "https://api.retellai.com/create-phone-call"
                headers = {"Authorization": f"Bearer {st.session_state.retell_api_key}", "Content-Type": "application/json"}
                body = {
                    "from_number": st.session_state.from_number,
                    "to_number": lead['phone'],
                    "override_agent_id": st.session_state.agent_id,
                    "retell_llm_dynamic_variables": {
                        "name": lead.get('name', ''),
                        "pitch": selected_pitch
                    }
                }
                resp = requests.post(url, headers=headers, json=body)
                resp.raise_for_status()
                call_id = resp.json().get('call_id', 'Unknown')
                st.success(f"Call initiated with pitch: {selected_pitch}")
                st.session_state.call_logs.append({
                    'phone': lead['phone'],
                    'name': lead.get('name', 'Unknown'),
                    'time': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'status': 'Initiated',
                    'call_id': call_id,
                    'pitch_used': selected_pitch
                })
                save_to_json("call_logs", st.session_state.call_logs)
            except Exception as e:
                st.error(f"Failed to initiate call: {str(e)}")

# Batch Call All (with A/B if enabled)
if st.button("Batch Call All"):
    if st.session_state.leads:
        def batch_call():
            for i, lead in enumerate(st.session_state.leads):
                if 'pitch' in lead:
                    selected_pitch = lead['pitch']
                    if pitch_mode == "A/B Testing" and ab_pitch_a and ab_pitch_b:
                        selected_pitch = random.choice([ab_pitch_a, ab_pitch_b])
                    try:
                        url = "https://api.retellai.com/create-phone-call"
                        headers = {"Authorization": f"Bearer {st.session_state.retell_api_key}", "Content-Type": "application/json"}
                        body = {
                            "from_number": st.session_state.from_number,
                            "to_number": lead['phone'],
                            "override_agent_id": st.session_state.agent_id,
                            "retell_llm_dynamic_variables": {
                                "name": lead.get('name', ''),
                                "pitch": selected_pitch
                            }
                        }
                        resp = requests.post(url, headers=headers, json=body)
                        resp.raise_for_status()
                        call_id = resp.json().get('call_id', 'Unknown')
                        st.session_state.call_logs.append({
                            'phone': lead['phone'],
                            'name': lead.get('name', 'Unknown'),
                            'time': time.strftime("%Y-%m-%d %H:%M:%S"),
                            'status': 'Initiated',
                            'call_id': call_id,
                            'pitch_used': selected_pitch
                        })
                        save_to_json("call_logs", st.session_state.call_logs)
                        time.sleep(st.session_state.batch_delay)
                    except Exception as e:
                        st.error(f"Failed to initiate call for lead {i+1}: {str(e)}")
            st.session_state.status = "Batch calls complete."
            st.rerun()
        
        threading.Thread(target=batch_call).start()

# Display Call Logs
if st.session_state.call_logs:
    st.subheader("Call Logs")
    log_df = pd.DataFrame(st.session_state.call_logs)
    st.dataframe(log_df)

# Detailed Reports (Conversion Status Tracking)
st.subheader("Call Analysis and Reports")
if st.button("Update Analyses"):
    updated = 0
    for log in st.session_state.call_logs:
        if 'call_id' in log and 'transcript' not in log:
            try:
                url = f"https://api.retellai.com/v2/get-call/{log['call_id']}"
                headers = {"Authorization": f"Bearer {st.session_state.retell_api_key}"}
                resp = requests.get(url, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                log['transcript'] = data.get('transcript', '')
                log['sentiment'] = data.get('call_analysis', {}).get('user_sentiment', 'Unknown')
                log['summary'] = data.get('call_analysis', {}).get('call_summary', '')
                log['success'] = data.get('call_analysis', {}).get('call_successful', False)
                updated += 1
            except Exception as e:
                st.error(f"Failed for {log['call_id']}: {str(e)}")
    save_to_json("call_logs", st.session_state.call_logs)
    st.success(f"Updated {updated} logs.")

# Post-Call Feedback Loop
st.subheader("Post-Call Feedback")
selected_log_index = st.selectbox("Select Call Log for Feedback", options=range(len(st.session_state.call_logs)), format_func=lambda i: f"{st.session_state.call_logs[i]['phone']} - {st.session_state.call_logs[i]['status']}")
if st.button("Analyze Transcript"):
    if st.session_state.call_logs:
        log = st.session_state.call_logs[selected_log_index]
        if 'transcript' in log:
            try:
                client = openai.OpenAI(api_key=st.session_state.openai_api_key)
                prompt = f"Analyze this transcript and suggest improvements: {log['transcript']}"
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "system", "content": "Analyze sales call transcript for improvements."}, {"role": "user", "content": prompt}]
                )
                feedback = response.choices[0].message.content
                st.text_area("Feedback", value=feedback, height=200)
            except Exception as e:
                st.error(f"Failed to analyze transcript: {str(e)}")

# Readiness Report
if st.button("Generate Readiness Report"):
    ready_leads = len([lead for lead in st.session_state.leads if 'pitch' in lead])
    total_leads = len(st.session_state.leads)
    st.write(f"Readiness: {ready_leads}/{total_leads} leads have pitches.")
    st.write(f"Agent Status: {'Ready' if st.session_state.agent_id else 'Not Set Up'}")

# Analytics Dashboard
if st.session_state.call_logs:
    df_logs = pd.DataFrame(st.session_state.call_logs)
    conversion_rate = df_logs['success'].mean() * 100 if 'success' in df_logs else 0
    st.write(f"Conversion Rate: {conversion_rate:.2f}%")
    sentiments = df_logs['sentiment'].value_counts() if 'sentiment' in df_logs else pd.Series()
    if not sentiments.empty:
        fig, ax = plt.subplots()
        sentiments.plot(kind='bar', ax=ax)
        st.pyplot(fig)

# Export Logs
if st.button("Export Logs to CSV"):
    if st.session_state.call_logs:
        df = pd.DataFrame(st.session_state.call_logs)
        csv = df.to_csv(index=False)
        st.download_button("Download Logs CSV", csv, "call_logs.csv", "text/csv")
    else:
        st.error("No logs to export.")

# AI Reformat Phone
def ai_reformat_phone(phone):
    if not st.session_state.openai_api_key:
        return None
    client = openai.OpenAI(api_key=st.session_state.openai_api_key)
    prompt = f"Format this phone number to E.164 format, assuming US if no country code: {phone}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a phone number formatter. Output only the formatted E.164 phone number, nothing else."}, {"role": "user", "content": prompt}]
    )
    try:
        formatted = response.choices[0].message.content.strip()
        return formatted
    except Exception as e:
        st.error(f"AI format error: {str(e)}")
        return None