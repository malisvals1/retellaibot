
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
from bs4 import BeautifulSoup  # For scraping (user must install if not present)
import multiprocessing  # For background scraping

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

# Input fields
st.subheader("API Keys and Config")
st.session_state.retell_api_key = st.text_input("Retell AI API Key", value=st.session_state.retell_api_key)
st.session_state.openai_api_key = st.text_input("OpenAI API Key", value=st.session_state.openai_api_key, type="password")
st.session_state.from_number = st.text_input("From Number (E.164 format)", value=st.session_state.from_number)

# Niche
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
st.subheader("AI Orchestrator for Scraping and Pitching")
if st.button("Generate AI Strategies"):
    if not st.session_state.openai_api_key or not st.session_state.niche or not st.session_state.product_description:
        st.error("OpenAI key, niche, and product description required.")
    else:
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {st.session_state.openai_api_key}", "Content-Type": "application/json"}
            body = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are a sales strategist. Output strict JSON."},
                    {"role": "user", "content": f"Given niche '{st.session_state.niche}', product '{st.session_state.product_description}', suggest: scraping_instructions (e.g., websites and queries), pitch_script, closing_tips. Output as JSON."}
                ]
            }
            resp = requests.post(url, headers=headers, json=body)
            resp.raise_for_status()
            ai_response = json.loads(resp.json()['choices'][0]['message']['content'])
            st.session_state.scraping_instructions = ai_response.get('scraping_instructions', '')
            st.session_state.custom_prompt = ai_response.get('pitch_script', st.session_state.custom_prompt)
            st.session_state.closing_tips = ai_response.get('closing_tips', '')
            st.text_area("Scraping Instructions", st.session_state.scraping_instructions)
            st.text_area("Closing Tips", st.session_state.closing_tips)
            st.success("Strategies generated.")
        except Exception as e:
            st.error(f"Failed: {str(e)}")

# Setup Agent (incorporate closing tips)
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
                "general_prompt": "You are a sales agent calling {{name}}. Your goal is to sell using the following pitch: {{pitch}}. Be engaging, ask questions, handle objections, and try to close the sale." + f" {st.session_state.closing_tips}"
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

# Scrape Leads
st.subheader("Scrape Leads")
st.session_state.scrape_url = st.text_input("Scrape URL (e.g., from AI instructions)", value=st.session_state.scrape_url)
if st.button("Scrape Leads Now"):
    if st.session_state.scrape_url:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(st.session_state.scrape_url, headers=headers)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            # Extract phones and names (simple heuristic)
            phones = []
            names = []
            for text in soup.find_all(text=True):
                phone_matches = re.findall(r'(\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})', text)
                if phone_matches:
                    phones.extend(phone_matches)
                if re.match(r'^[A-Z][a-z]+\s[A-Z][a-z]+$', text.strip()):
                    names.append(text.strip())
            new_leads = []
            phones = list(set(phones))  # Dedupe
            for i, phone in enumerate(phones):
                fixed = ai_reformat_phone(phone)
                if fixed:
                    name = names[i % len(names)] if names else "Unknown"
                    new_leads.append({"phone": fixed, "name": name, "info": st.session_state.niche})
            st.session_state.leads.extend(new_leads)
            save_to_json("leads", st.session_state.leads)
            st.success(f"Added {len(new_leads)} leads from scrape.")
        except Exception as e:
            st.error(f"Scraping failed: {str(e)}")
    else:
        st.error("Provide a scrape URL.")

# Background Scraping
if st.checkbox("Enable Unlimited Background Scraping"):
    def background_scraper(url):
        while True:
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                resp = requests.get(url, headers=headers)
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
                        new_leads.append({"phone": fixed, "name": name, "info": st.session_state.niche})
                st.session_state.leads.extend(new_leads)
                save_to_json("leads", st.session_state.leads)
                time.sleep(3600)  # 1 hour throttle to avoid bans
            except:
                time.sleep(3600)

    if st.session_state.background_scraping_process is None or not st.session_state.background_scraping_process.is_alive():
        st.session_state.background_scraping_process = multiprocessing.Process(target=background_scraper, args=(st.session_state.scrape_url,))
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
            st.session_state.leads = df.to_dict(orient='records')
            save_to_json("leads", st.session_state.leads)
            st.success(f"Loaded {len(st.session_state.leads)} leads.")
    except Exception as e:
        st.error(f"Failed to load CSV: {str(e)}")

# Display Leads
if st.session_state.leads:
    st.subheader("Leads")
    lead_df = pd.DataFrame(st.session_state.leads)
    st.dataframe(lead_df)

# Qualify Leads
st.subheader("Qualify Leads")
if st.button("Qualify All Leads"):
    for lead in st.session_state.leads:
        info = lead.get('info', '')
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {st.session_state.openai_api_key}", "Content-Type": "application/json"}
            body = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "Score lead intent from 1-10."},
                    {"role": "user", "content": f"Score this lead for niche {st.session_state.niche}: {info}"}
                ]
            }
            resp = requests.post(url, headers=headers, json=body)
            score = int(resp.json()['choices'][0]['message']['content'].strip())
            lead['score'] = score
        except:
            lead['score'] = 0
    save_to_json("leads", st.session_state.leads)
    st.success("Leads qualified.")

# Generate Pitch for Selected Lead
st.subheader("Generate Pitch for Selected Lead")
selected_lead_index = st.selectbox("Select Lead", options=range(len(st.session_state.leads)), format_func=lambda i: f"{st.session_state.leads[i].get('name', 'Unknown')} - {st.session_state.leads[i]['phone']}" if 'leads' in st.session_state else "No leads loaded")
pitch_mode = st.selectbox("Pitch Mode", ["Auto-Pitch", "Niche-Based", "Override"])
if st.button("Generate Pitch for Selected"):
    if 'leads' in st.session_state and st.session_state.leads:
        lead = st.session_state.leads[selected_lead_index]
        name = lead.get('name', 'the customer')
        info = lead.get('info', 'no specific information available')
        product = st.session_state.product_description or "our amazing product/service"
        
        custom_prompt = st.session_state.custom_prompt
        user_content = ""
        if custom_prompt:
            try:
                user_content = custom_prompt.format(name=name, info=info, product=product)
            except KeyError:
                st.error("Custom prompt should use {name}, {info}, {product} if needed.")
        else:
            user_content = f"Come up with the very best thing to sell to {name} based on {info}. The product is {product}. Generate a compelling pitch script for a phone call."
        
        if pitch_mode == "Niche-Based":
            user_content += f" Tailor for niche: {st.session_state.niche}."
        # For Auto-Pitch, it's handled by dynamic variables in call initiation
        # For Override, user can edit after generation
        
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
            st.text_area("Generated Pitch (Editable)", value=pitch, height=200, key="pitch_display_selected")
            st.success("Pitch generated.")
        except Exception as e:
            st.error(f"Failed to generate pitch: {str(e)}")

# Generate Pitches for All
if st.button("Generate Pitches for All"):
    if 'leads' in st.session_state and st.session_state.leads:
        def batch_gen():
            for i, lead in enumerate(st.session_state.leads):
                name = lead.get('name', 'the customer')
                info = lead.get('info', 'no specific information available')
                product = st.session_state.product_description or "our amazing product/service"
                
                custom_prompt = st.session_state.custom_prompt
                if custom_prompt:
                    try:
                        user_content = custom_prompt.format(name=name, info=info, product=product)
                    except KeyError:
                        st.error("Custom prompt should use {name}, {info}, {product} if needed.")
                        return
                else:
                    user_content = f"Come up with the very best thing to sell to {name} based on {info}. The product is {product}. Generate a compelling pitch script for a phone call."
                
                if pitch_mode == "Niche-Based":
                    user_content += f" Tailor for niche: {st.session_state.niche}."
                
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
                    st.session_state.status = f"Generated pitch for lead {i+1}/{len(st.session_state.leads)}"
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to generate pitch for lead {i+1}: {str(e)}")
                    return
            st.session_state.status = "Batch pitch generation complete."
            st.rerun()
        
        threading.Thread(target=batch_gen).start()

# Initiate Call for Selected
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
                save_to_json("call_logs", st.session_state.call_logs)
            except Exception as e:
                st.error(f"Failed to initiate call: {str(e)}")

# Batch Call All
if st.button("Batch Call All"):
    if 'leads' in st.session_state and st.session_state.leads:
        def batch_call():
            for i, lead in enumerate(st.session_state.leads):
                if 'pitch' in lead:
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
                        st.session_state.call_logs.append({
                            'phone': lead['phone'],
                            'name': lead.get('name', 'Unknown'),
                            'time': time.strftime("%Y-%m-%d %H:%M:%S"),
                            'status': 'Initiated',
                            'call_id': call_id
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

# Fetch Call Analyses
st.subheader("Call Analysis and Reports")
if st.button("Update Analyses"):
    updated = 0
    for log in st.session_state.call_logs:
        if 'call_id' in log and 'transcript' not in log:
            try:
                url = f"https://api.retellai.com/get-call/{log['call_id']}"
                headers = {"Authorization": f"Bearer {st.session_state.retell_api_key}"}
                resp = requests.get(url, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                log['transcript'] = data.get('transcript', '')
                log['sentiment'] = data.get('user_sentiment', 'Unknown')
                log['summary'] = data.get('call_summary', '')
                log['success'] = data.get('call_successful', False)
                updated += 1
            except Exception as e:
                st.error(f"Failed for {log['call_id']}: {str(e)}")
    save_to_json("call_logs", st.session_state.call_logs)
    st.success(f"Updated {updated} logs.")

# Post-Call Feedback
st.subheader("Post-Call Feedback")
selected_log_index = st.selectbox("Select Call Log", options=range(len(st.session_state.call_logs)), format_func=lambda i: f"{st.session_state.call_logs[i]['phone']} - {st.session_state.call_logs[i]['status']}")
if st.button("Analyze Transcript"):
    if st.session_state.call_logs:
        log = st.session_state.call_logs[selected_log_index]
        if 'transcript' in log:
            try:
                url = "https://api.openai.com/v1/chat/completions"
                headers = {"Authorization": f"Bearer {st.session_state.openai_api_key}", "Content-Type": "application/json"}
                body = {
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": "Analyze sales call transcript for improvements."},
                        {"role": "user", "content": f"Analyze this transcript and suggest improvements: {log['transcript']}"}
                    ]
                }
                resp = requests.post(url, headers=headers, json=body)
                resp.raise_for_status()
                feedback = resp.json()['choices'][0]['message']['content']
                st.text_area("Feedback", value=feedback, height=200)
            except Exception as e:
                st.error(f"Failed to analyze transcript: {str(e)}")

# Readiness Report
if st.button("Generate Readiness Report"):
    ready_leads = len([lead for lead in st.session_state.leads if 'pitch' in lead])
    total_leads = len(st.session_state.leads)
    st.write(f"Readiness: {ready_leads}/{total_leads} leads have pitches.")
    st.write(f"Agent Status: {'Ready' if st.session_state.agent_id else 'Not Set Up'}")
    # Could add email, but skip for now

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
        return None

# Notes for scalability (commented)
# For scaling, consider using RQ with Redis for background tasks
# For multi-tenancy, use streamlit-authenticator and per-user DB
# Integrate Stripe for subscriptions if monetizing
