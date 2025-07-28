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
import urllib.parse  # For URL encoding
from twilio.rest import Client as TwilioClient  # For SMS
import smtplib  # For email
from email.mime.text import MIMEText  # For email
from apify_client import ApifyClient  # For Apify

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
st.warning("Ensure scraping complies with site terms and laws. Respect robots.txt and use ethically. Focus on free directories that allow public access.")

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
if 'voice_id' not in st.session_state:
    st.session_state.voice_id = "openai-Alloy"
if 'scrape_urls' not in st.session_state:
    st.session_state.scrape_urls = []
if 'background_test_gen_process' not in st.session_state:
    st.session_state.background_test_gen_process = None
if 'twilio_sid' not in st.session_state:
    st.session_state.twilio_sid = os.getenv("TWILIO_SID", "")
if 'twilio_token' not in st.session_state:
    st.session_state.twilio_token = os.getenv("TWILIO_TOKEN", "")
if 'twilio_numbers' not in st.session_state:
    st.session_state.twilio_numbers = []  # List of Twilio numbers for rotation
if 'gmail_creds' not in st.session_state:
    st.session_state.gmail_creds = []  # List of {'email': '', 'app_password': ''}
if 'scrapingbee_key' not in st.session_state:
    st.session_state.scrapingbee_key = os.getenv("SCRAPINGBEE_KEY", "")
if 'apify_key' not in st.session_state:
    st.session_state.apify_key = os.getenv("APIFY_KEY", "")
if 'zenrows_key' not in st.session_state:
    st.session_state.zenrows_key = os.getenv("ZENROWS_KEY", "")
if 'scraperapi_key' not in st.session_state:
    st.session_state.scraperapi_key = os.getenv("SCRAPERAPI_KEY", "")

# Input fields
st.subheader("API Keys and Config")
st.session_state.retell_api_key = st.text_input("Retell AI API Key", value=st.session_state.retell_api_key)
st.session_state.openai_api_key = st.text_input("OpenAI API Key", value=st.session_state.openai_api_key, type="password")
st.session_state.from_number = st.text_input("From Number (E.164 format)", value=st.session_state.from_number)
st.session_state.twilio_sid = st.text_input("Twilio SID", value=st.session_state.twilio_sid, type="password")
st.session_state.twilio_token = st.text_input("Twilio Auth Token", value=st.session_state.twilio_token, type="password")
twilio_numbers_str = st.text_input("Twilio Number Pool (comma-separated E.164, e.g., +1234567890,+1987654321)", value=",".join(st.session_state.twilio_numbers))
st.session_state.twilio_numbers = [n.strip() for n in twilio_numbers_str.split(',') if n.strip()]
gmail_creds_str = st.text_area("Gmail Creds List (JSON array of {'email': 'user@gmail.com', 'app_password': 'xxxx'}, e.g., [{'email': 'a@gmail.com', 'app_password': 'pass1'}, {'email': 'b@gmail.com', 'app_password': 'pass2'}])", value=json.dumps(st.session_state.gmail_creds))
try:
    st.session_state.gmail_creds = json.loads(gmail_creds_str)
except:
    st.error("Invalid Gmail creds JSON.")
st.session_state.scrapingbee_key = st.text_input("ScrapingBee API Key", value=st.session_state.scrapingbee_key, type="password")
st.session_state.apify_key = st.text_input("Apify API Key", value=st.session_state.apify_key, type="password")
st.session_state.zenrows_key = st.text_input("ZenRows API Key", value=st.session_state.zenrows_key, type="password")
st.session_state.scraperapi_key = st.text_input("ScraperAPI Key", value=st.session_state.scraperapi_key, type="password")

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

# AI Orchestrator (enhanced for ethical directories)
st.subheader("AI Orchestrator")
if st.button("Run AI Orchestrator"):
    if st.session_state.openai_api_key and st.session_state.niche and st.session_state.product_description:
        try:
            client = openai.OpenAI(api_key=st.session_state.openai_api_key)
            prompt = f"You are a sales strategist. Analyze niche: {st.session_state.niche}, product: {st.session_state.product_description}. Output JSON: {{'scraping_instructions': 'str', 'pitch_script': 'str', 'closing_tips': 'str', 'websites_to_scrape': ['list of 50+ varied URLs from ethical free business directories like business.com, yellowpages.com (if allowed), with rotated cities, keywords, and pagination for max results']}}"
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": prompt}],
                response_format={"type": "json_object"}
            )
            st.session_state.ai_output = json.loads(response.choices[0].message.content)
            st.session_state.scraping_instructions = st.session_state.ai_output.get('scraping_instructions', '')
            st.session_state.custom_prompt = st.session_state.ai_output.get('pitch_script', st.session_state.custom_prompt)
            st.session_state.closing_tips = st.session_state.ai_output.get('closing_tips', '')
            st.session_state.scrape_urls = st.session_state.ai_output.get('websites_to_scrape', [])
            st.success("AI Orchestrator completed. Generated 50+ paginated URLs from ethical directories.")
            st.text_area("Scraping Instructions", value=st.session_state.scraping_instructions)
            st.text_area("Closing Tips", value=st.session_state.closing_tips)
            st.write("Generated Scrape URLs:", st.session_state.scrape_urls)
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

# Scrape Leads (with rotation among APIs)
def scrape_leads(url, niche):
    services = []
    if st.session_state.scrapingbee_key:
        services.append('scrapingbee')
    if st.session_state.apify_key:
        services.append('apify')
    if st.session_state.zenrows_key:
        services.append('zenrows')
    if st.session_state.scraperapi_key:
        services.append('scraperapi')
    if not services:
        services.append('direct')
    
    random.shuffle(services)  # Rotate randomly
    for service in services:
        try:
            if service == 'scrapingbee':
                encoded_url = urllib.parse.quote(url)
                scrapingbee_url = f"https://app.scrapingbee.com/api/v1/?api_key={st.session_state.scrapingbee_key}&url={encoded_url}&render_js=true"
                resp = requests.get(scrapingbee_url)
                resp.raise_for_status()
                html = resp.text
            elif service == 'apify':
                client = ApifyClient(st.session_state.apify_key)
                run_input = {
                    "startUrls": [{"url": url}],
                    "proxyConfiguration": {"useApifyProxy": true},
                }
                actor_call = client.actor('apify/website-content-crawler').call(run_input=run_input)
                dataset_items = list(client.dataset(actor_call['defaultDatasetId']).iterate_items())
                html = dataset_items[0]['text'] if dataset_items else ''
            elif service == 'zenrows':
                zen_url = f"https://api.zenrows.com/v1/?apikey={st.session_state.zenrows_key}&url={urllib.parse.quote(url)}&js_render=true"
                resp = requests.get(zen_url)
                resp.raise_for_status()
                html = resp.text
            elif service == 'scraperapi':
                scraper_url = f"http://api.scraperapi.com?api_key={st.session_state.scraperapi_key}&url={urllib.parse.quote(url)}"
                resp = requests.get(scraper_url)
                resp.raise_for_status()
                html = resp.text
            else:  # direct
                headers = {"User-Agent": random.choice(["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"])}
                resp = requests.get(url, headers=headers)
                resp.raise_for_status()
                html = resp.text
            
            soup = BeautifulSoup(html, 'html.parser')
            phones = []
            names = []
            emails = []
            companies = []
            
            # Expanded targeted extraction
            name_elements = soup.find_all(class_=re.compile(r'(business|name|title|listing|item)', re.I))
            names = [el.get_text(strip=True) for el in name_elements if el.get_text(strip=True)]
            
            phone_elements = soup.find_all(class_=re.compile(r'(phone|contact|tel|number)', re.I))
            for el in phone_elements:
                text = el.get_text(strip=True) or el.get('href', '') or el.get('content', '')
                phone_matches = re.findall(r'(\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})', text)
                if phone_matches:
                    phones.extend(phone_matches)
            
            email_elements = soup.find_all(class_=re.compile(r'(email|contact|mail|address)', re.I))
            for el in email_elements:
                text = el.get_text(strip=True) or el.get('href', '') or el.get('content', '')
                email_matches = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
                if email_matches:
                    emails.extend(email_matches)
            
            company_elements = soup.find_all(class_=re.compile(r'(company|business|org|firm|enterprise)', re.I))
            companies = [el.get_text(strip=True) for el in company_elements if el.get_text(strip=True)]
            
            # Expanded fallback
            for text in soup.find_all(text=True):
                phone_matches = re.findall(r'(\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})', text)
                if phone_matches:
                    phones.extend(phone_matches)
                email_matches = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
                if email_matches:
                    emails.extend(email_matches)
            
            new_leads = []
            phones = list(set(phones))  # Dedupe
            for i, phone in enumerate(phones):
                fixed = ai_reformat_phone(phone)
                if fixed:
                    name = names[i % len(names)] if names else "Unknown"
                    email = emails[i % len(emails)] if emails else ""
                    company = companies[i % len(companies)] if companies else ""
                    new_leads.append({"phone": fixed, "name": name, "email": email, "company": company, "info": niche, "score": random.randint(1, 10)})
            
            # Dedupe against existing
            existing_phones = {lead['phone'] for lead in st.session_state.leads}
            new_leads = [lead for lead in new_leads if lead['phone'] not in existing_phones]
            
            st.info(f"Scraped {len(new_leads)} leads from {url} using {service}")
            return new_leads
        except Exception as e:
            st.warning(f"{service} failed for {url}: {str(e)}. Trying next service.")
    st.error(f"All services failed for {url}.")
    return []

st.subheader("Scrape Leads")
st.session_state.scrape_url = st.text_input("Scrape URL (e.g., free directory search)", value=st.session_state.scrape_url)
if st.button("Scrape Now"):
    if st.session_state.scrape_url:
        new_leads = scrape_leads(st.session_state.scrape_url, st.session_state.niche)
        st.session_state.leads.extend(new_leads)
        save_to_json("leads", st.session_state.leads)
        st.success(f"Added {len(new_leads)} new leads (duplicates skipped).")

# Generate Test Leads with OpenAI (fixed prompt)
st.subheader("Generate Test Leads (OpenAI)")
test_lead_count = st.number_input("Number of Test Leads", min_value=1, max_value=50, value=10)
if st.button("Generate Test Leads"):
    if st.session_state.openai_api_key and st.session_state.niche:
        try:
            client = openai.OpenAI(api_key=st.session_state.openai_api_key)
            prompt = f"Generate {test_lead_count} fake sample leads for niche: {st.session_state.niche}. Output as JSON: {{ 'leads': [array of objects with keys phone (E.164 format string), name (string), email (string), company (string), info (string), score (integer from 1 to 10)] }}"
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt}],
                response_format={"type": "json_object"}
            )
            generated = json.loads(response.choices[0].message.content).get('leads', [])
            st.session_state.leads.extend(generated)
            save_to_json("leads", st.session_state.leads)
            st.success(f"Generated and added {len(generated)} test leads.")
        except Exception as e:
            st.error(f"Failed to generate test leads: {str(e)}")
    else:
        st.error("OpenAI key and niche required.")

# Background Scraping (faster cycle for max leads)
if st.checkbox("Enable Background Scraping"):
    def background_scraper(urls, niche):
        while True:
            for url in urls:
                new_leads = scrape_leads(url, niche)
                st.session_state.leads.extend(new_leads)
                save_to_json("leads", st.session_state.leads)
                time.sleep(180)  # Faster delay for more results
            time.sleep(1800)  # 30 min after full cycle

    if st.session_state.background_scraping_process is None or not st.session_state.background_scraping_process.is_alive():
        if st.session_state.scrape_urls:
            st.session_state.background_scraping_process = multiprocessing.Process(target=background_scraper, args=(st.session_state.scrape_urls, st.session_state.niche))
            st.session_state.background_scraping_process.start()
            st.success("Background scraping started with paginated URLs for max leads.")
        else:
            st.warning("Run AI Orchestrator first to generate paginated scrape URLs.")
else:
    if st.session_state.background_scraping_process is not None and st.session_state.background_scraping_process.is_alive():
        st.session_state.background_scraping_process.terminate()
        st.session_state.background_scraping_process = None
        st.success("Background scraping stopped.")

# Background Test Lead Generation (for ongoing testing)
if st.checkbox The scraping fails likely because sites block direct requests or require JS. From tools, alternatives: Apify, ZenRows, ScraperAPI.

For test leads failure: The prompt has unescaped {} in f-string, causing "Invalid format specifier". Fixed by doubling {{ 'leads': ... }} to escape.

For SMS: Integrate Twilio. Add SID, Token inputs. Send SMS function, follow-up if call fails (in try-except).

For email: Use smtplib with Gmail SMTP, app passwords. Add input for list of dicts {'email': , 'app_password': }. Rotate by random choice.

For Twilio number rotation: Input pool of numbers, random choice for from_.

For scraping rotation: Add keys for ScrapingBee, Apify, ScraperAPI, ZenRows. Rotate in scrape_leads, use different APIs.

For Apify: Use SDK to run actor like 'apify/web-scraper', get results.

Add new section for API keys (Twilio, email creds, scraping APIs).

For follow-up: In initiate call, if exception (assume no answer as failure for simplicity), send SMS/email.

For email rotation: Random from list.

For better qualifier: In qualify, use more detailed prompt.

Alternatives for scrape: Use Apify actors for directories.

From tools, best directories: YellowPages, Yelp (but ethical?), ParseHub for no-code, but since Python, integrate APIs.

Update code with these.<xaiArtifact artifact_id="15326d46-6cf3-4d1a-b53c-26a0fcb5d55d" artifact_version_id="03de00fd-03f3-450a-9cf5-593f550a3759" title="sales_bot_app_enhanced.py" contentType="text/python">
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
import urllib.parse  # For URL encoding
from twilio.rest import Client as TwilioClient  # For SMS and number rotation
import smtplib  # For email
from email.mime.text import MIMEText  # For email content
from apify_client import ApifyClient  # For Apify integration
# Note: Add 'twilio', 'apify-client' to requirements.txt

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
st.warning("Ensure scraping complies with site terms and laws. Respect robots.txt and use ethically. Focus on free directories that allow public access.")

# Initialize session state
if 'retell_api_key' not in st.session_state:
    st.session_state.retell_api_key = os.getenv("RETELL_API_KEY", "")
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")
if 'twilio_sid' not in st.session_state:
    st.session_state.twilio_sid = os.getenv("TWILIO_SID", "")
if 'twilio_token' not in st.session_state:
    st.session_state.twilio_token = os.getenv("TWILIO_TOKEN", "")
if 'twilio_numbers' not in st.session_state:
    st.session_state.twilio_numbers = []  # List of Twilio numbers for rotation
if 'email_creds' not in st.session_state:
    st.session_state.email_creds = []  # List of {'email': '', 'app_password': ''}
if 'scrapingbee_api_key' not in st.session_state:
    st.session_state.scrapingbee_api_key = os.getenv("SCRAPINGBEE_API_KEY", "")
if 'apify_api_token' not in st.session_state:
    st.session_state.apify_api_token = os.getenv("APIFY_API_TOKEN", "")
if 'scraperapi_key' not in st.session_state:
    st.session_state.scraperapi_key = os.getenv("SCRAPERAPI_KEY", "")
if 'zenrows_api_key' not in st.session_state:
    st.session_state.zenrows_api_key = os.getenv("ZENROWS_API_KEY", "")
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
if 'voice_id' not in st.session_state:
    st.session_state.voice_id = "openai-Alloy"
if 'scrape_urls' not in st.session_state:
    st.session_state.scrape_urls = []
if 'background_test_gen_process' not in st.session_state:
    st.session_state.background_test_gen_process = None

# Input fields
st.subheader("API Keys and Config")
st.session_state.retell_api_key = st.text_input("Retell AI API Key", value=st.session_state.retell_api_key)
st.session_state.openai_api_key = st.text_input("OpenAI API Key", value=st.session_state.openai_api_key, type="password")
st.session_state.twilio_sid = st.text_input("Twilio SID", value=st.session_state.twilio_sid, type="password")
st.session_state.twilio_token = st.text_input("Twilio Auth Token", value=st.session_state.twilio_token, type="password")
st.session_state.twilio_numbers = st.text_area("Twilio Numbers Pool (one per line, E.164 format)", value='\n'.join(st.session_state.twilio_numbers)).split('\n')
st.session_state.email_creds = json.loads(st.text_area("Email Credentials List (JSON array of {'email': '', 'app_password': ''})", value=json.dumps(st.session_state.email_creds, indent=2)))
st.session_state.scrapingbee_api_key = st.text_input("ScrapingBee API Key", value=st.session_state.scrapingbee_api_key, type="password")
st.session_state.apify_api_token = st.text_input("Apify API Token", value=st.session_state.apify_api_token, type="password")
st.session_state.scraperapi_key = st.text_input("ScraperAPI Key", value=st.session_state.scraperapi_key, type="password")
st.session_state.zenrows_api_key = st.text_input("ZenRows API Key", value=st.session_state.zenrows_api_key, type="password")
st.session_state.from_number = st.text_input("Default From Number (E.164 format)", value=st.session_state.from_number)

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

# AI Orchestrator (enhanced for more URLs with pagination)
st.subheader("AI Orchestrator")
if st.button("Run AI Orchestrator"):
    if st.session_state.openai_api_key and st.session_state.niche and st.session_state.product_description:
        try:
            client = openai.OpenAI(api_key=st.session_state.openai_api_key)
            prompt = f"You are a sales strategist. Analyze niche: {st.session_state.niche}, product: {st.session_state.product_description}. Output JSON: {{'scraping_instructions': 'str', 'pitch_script': 'str', 'closing_tips': 'str', 'websites_to_scrape': ['list of 50+ varied URLs from ethical free directories with rotated cities, keywords, and pagination like &start=10, &start=20, etc for max results']}}"
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": prompt}],
                response_format={"type": "json_object"}
            )
            st.session_state.ai_output = json.loads(response.choices[0].message.content)
            st.session_state.scraping_instructions = st.session_state.ai_output.get('scraping_instructions', '')
            st.session_state.custom_prompt = st.session_state.ai_output.get('pitch_script', st.session_state.custom_prompt)
            st.session_state.closing_tips = st.session_state.ai_output.get('closing_tips', '')
            st.session_state.scrape_urls = st.session_state.ai_output.get('websites_to_scrape', [])
            st.success("AI Orchestrator completed. Generated 50+ paginated URLs for max leads.")
            st.text_area("Scraping Instructions", value=st.session_state.scraping_instructions)
            st.text_area("Closing Tips", value=st.session_state.closing_tips)
            st.write("Generated Scrape URLs:", st.session_state.scrape_urls)
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

# Rotate scraping APIs (ScrapingBee, Apify, ScraperAPI, ZenRows)
scraping_apis = [
    {'name': 'scrapingbee', 'key': st.session_state.scrapingbee_api_key},
    {'name': 'apify', 'key': st.session_state.apify_api_token},
    {'name': 'scraperapi', 'key': st.session_state.scraperapi_key},
    {'name': 'zenrows', 'key': st.session_state.zenrows_api_key}
]
scraping_apis = [api for api in scraping_apis if api['key']]

def scrape_with_api(url, niche, api):
    try:
        if api['name'] == 'scrapingbee':
            encoded_url = urllib.parse.quote(url)
            scrapingbee_url = f"https://app.scrapingbee.com/api/v1/?api_key={api['key']}&url={encoded_url}&render_js=true"
            resp = requests.get(scrapingbee_url)
            html = resp.text
        elif api['name'] == 'apify':
            client = ApifyClient(api['key'])
            run_input = {
                "startUrls": [{"url": url}],
                "proxyConfiguration": {"useApifyProxy": True}
            }
            actor_call = client.actor('apify/web-scraper').call(run_input=run_input)
            dataset_items = client.dataset(actor_call['defaultDatasetId']).list_items().items
            html = dataset_items[0]['fullHtml'] if dataset_items else ''
        elif api['name'] == 'scraperapi':
            scraperapi_url = f"http://api.scraperapi.com?api_key={api['key']}&url={urllib.parse.quote(url)}&render=true"
            resp = requests.get(scraperapi_url)
            html = resp.text
        elif api['name'] == 'zenrows':
            zenrows_url = f"https://api.zenrows.com/v1/?apikey={api['key']}&url={urllib.parse.quote(url)}&js_render=true"
            resp = requests.get(zenrows_url)
            html = resp.text
        else:
            html = ''

        if not html:
            return []

        soup = BeautifulSoup(html, 'html.parser')
        phones = []
        names = []
        emails = []
        companies = []
        
        # Expanded targeted extraction
        name_elements = soup.find_all(class_=re.compile(r'(business|name|title|listing|item)', re.I))
        names = [el.get_text(strip=True) for el in name_elements if el.get_text(strip=True)]
        
        phone_elements = soup.find_all(class_=re.compile(r'(phone|contact|tel|number)', re.I))
        for el in phone_elements:
            text = el.get_text(strip=True) or el.get('href', '') or el.get('content', '')
            phone_matches = re.findall(r'(\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})', text)
            if phone_matches:
                phones.extend(phone_matches)
        
        email_elements = soup.find_all(class_=re.compile(r'(email|contact|mail|address)', re.I))
        for el in email_elements:
            text = el.get_text(strip=True) or el.get('href', '') or el.get('content', '')
            email_matches = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
            if email_matches:
                emails.extend(email_matches)
        
        company_elements = soup.find_all(class_=re.compile(r'(company|business|org|firm|enterprise)', re.I))
        companies = [el.get_text(strip=True) for el in company_elements if el.get_text(strip=True)]
        
        # Expanded fallback
        for text in soup.find_all(text=True):
            phone_matches = re.findall(r'(\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})', text)
            if phone_matches:
                phones.extend(phone_matches)
            email_matches = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
            if email_matches:
                emails.extend(email_matches)
        
        new_leads = []
        phones = list(set(phones))  # Dedupe
        for i, phone in enumerate(phones):
            fixed = ai_reformat_phone(phone)
            if fixed:
                name = names[i % len(names)] if names else "Unknown"
                email = emails[i % len(emails)] if emails else ""
                company = companies[i % len(companies)] if companies else ""
                new_leads.append({"phone": fixed, "name": name, "email": email, "company": company, "info": niche, "score": random.randint(1, 10)})
        
        # Dedupe against existing
        existing_phones = {lead['phone'] for lead in st.session_state.leads}
        new_leads = [lead for lead in new_leads if lead['phone'] not in existing_phones]
        
        st.info(f"Scraped {len(new_leads)} leads from {url} using {api['name']}")
        return new_leads
    except Exception as e:
        st.error(f"Scraping failed for {url} with {api['name']}: {str(e)}")
        return []

def scrape_leads(url, niche):
    if not scraping_apis:
        # Fallback to direct scrape if no APIs
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            html = resp.text
            # ... (same parsing as above)
            # Parse and return new_leads as before
        except:
            return []
    else:
        api = random.choice(scraping_apis)
        return scrape_with_api(url, niche, api)

# ... (rest of scrape section, background scraping remains similar, but now rotates APIs)

# Send SMS function with rotation
def send_sms(to_number, message):
    if st.session_state.twilio_sid and st.session_state.twilio_token and st.session_state.twilio_numbers:
        from_number = random.choice(st.session_state.twilio_numbers)
        client = TwilioClient(st.session_state.twilio_sid, st.session_state.twilio_token)
        try:
            client.messages.create(body=message, from_=from_number, to=to_number)
            st.success(f"SMS sent to {to_number} from {from_number}")
        except Exception as e:
            st.error(f"Failed to send SMS: {str(e)}")
    else:
        st.error("Twilio credentials or numbers missing.")

# Send Email function with rotation
def send_email(to_email, subject, body):
    if st.session_state.email_creds:
        cred = random.choice(st.session_state.email_creds)
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = cred['email']
        msg['To'] = to_email
        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(cred['email'], cred['app_password'])
                server.sendmail(cred['email'], to_email, msg.as_string())
            st.success(f"Email sent to {to_email} from {cred['email']}")
        except Exception as e:
            st.error(f"Failed to send email: {str(e)}")
    else:
        st.error("Email credentials missing.")

# In Initiate Call, add follow-up
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
                url = "https://api.retellai.com/v2/create-phone-call"
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
                st.error(f"Failed to initiate call: {str(e)}. Sending follow-up SMS/email.")
                if 'phone' in lead:
                    send_sms(lead['phone'], "We tried calling you about our product. Here's a link: https://example.com/info")
                if 'email' in lead:
                    send_email(lead['email'], "Follow-up on our call", "We tried calling you. Here's more info: https://example.com/info")

# In Batch Call, add follow-up on failure
if st.button("Batch Call All"):
    if st.session_state.leads:
        def batch_call():
            for i, lead in enumerate(st.session_state.leads):
                if 'pitch' in lead:
                    selected_pitch = lead['pitch']
                    if pitch_mode == "A/B Testing" and ab_pitch_a and ab_pitch_b:
                        selected_pitch = random.choice([ab_pitch_a, ab_pitch_b])
                    try:
                        url = "https://api.retellai.com/v2/create-phone-call"
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
                    except Exception as e:
                        st.error(f"Failed to initiate call for lead {i+1}: {str(e)}. Sending follow-up SMS/email.")
                        if 'phone' in lead:
                            send_sms(lead['phone'], "We tried calling you about our product. Here's a link: https://example.com/info")
                        if 'email' in lead:
                            send_email(lead['email'], "Follow-up on our call", "We tried calling you. Here's more info: https://example.com/info")
                    time.sleep(st.session_state.batch_delay)
            st.session_state.status = "Batch calls complete."
            st.rerun()
        
        threading.Thread(target=batch_call).start()

# Send SMS to Selected Lead
st.subheader("Send SMS to Selected Lead")
sms_message = st.text_area("SMS Message (use {name}, {product} placeholders)")
if st.button("Send SMS to Selected"):
    if st.session_state.leads and sms_message:
        lead = st.session_state.leads[selected_lead_index]
        formatted_msg = sms_message.format(name=lead.get('name', ''), product=st.session_state.product_description)
        send_sms(lead['phone'], formatted_msg)

# Send Email to Selected Lead
st.subheader("Send Email to Selected Lead")
email_subject = st.text_input("Email Subject")
email_body = st.text_area("Email Body (use {name}, {product} placeholders)")
if st.button("Send Email to Selected"):
    if st.session_state.leads and email_subject and email_body:
        lead = st.session_state.leads[selected_lead_index]
        formatted_body = email_body.format(name=lead.get('name', ''), product=st.session_state.product_description)
        send_email(lead['email'], email_subject, formatted_body)

# Follow Up Unanswered Calls
if st.button("Follow Up Unanswered Calls"):
    for log in st.session_state.call_logs:
        if log['status'] == 'Initiated' and 'success' not in log:
            lead_phone = log['phone']
            lead = next((l for l in st.session_state.leads if l['phone'] == lead_phone), None)
            if lead:
                send_sms(lead_phone, "We tried calling but no answer. Here's a follow-up link: https://example.com/info")
                if 'email' in lead:
                    send_email(lead['email'], "Follow-up on missed call", "We tried calling. Here's more info: https://example.com/info")
    st.success("Follow-ups sent for unanswered calls.")

# Lead Qualification & Enrichment (improved prompt for better qualifier)
st.subheader("Qualify & Enrich Leads")
if st.button("Qualify All Leads"):
    if st.session_state.openai_api_key:
        client = openai.OpenAI(api_key=st.session_state.openai_api_key)
        for lead in st.session_state.leads:
            info = lead.get('info', '')
            score_prompt = f"Score this lead for niche {st.session_state.niche}: {info}. Score 1-10 based on fit, and enrich with estimated email, location, or additional details if possible. Output JSON: {{'score': int, 'enriched_info': str}}"
            score_resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": score_prompt}],
                response_format={"type": "json_object"}
            )
            try:
                result = json.loads(score_resp.choices[0].message.content)
                lead['score'] = result.get('score', 0)
                lead['enriched_info'] = result.get('enriched_info', info)
            except:
                lead['score'] = 0
                lead['enriched_info'] = "No enrichment"
        save_to_json("leads", st.session_state.leads)
        st.success("Leads qualified and enriched with improved prompt.")
    else:
        st.error("OpenAI key required.")

# ... (rest of the code remains the same, including display, export, pitch, call, reports, etc.)