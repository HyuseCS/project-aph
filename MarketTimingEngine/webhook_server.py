from flask import Flask, Response, request, jsonify, current_app
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
import requests, os, hmac, hashlib, json, threading
import httpx

from MarketTiming import process_market_trends, handle_farmer_sms
import xgboost as xgb

app = Flask(__name__)
load_dotenv()

SMSGATE_PHONE_IP = os.getenv("SMSGATE_PHONE_IP")
SMSGATE_API_USERNAME = os.getenv("SMSGATE_API_USERNAME")
SMSGATE_API_PASSWORD = os.getenv("SMSGATE_API_PASSWORD")
SIGNING_KEY = os.getenv("SIGNING_KEY")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

app.config["SMS_GATE_API_USERNAME"] = SMSGATE_API_USERNAME
app.config["SMS_GATE_API_PASSWORD"] = SMSGATE_API_PASSWORD
app.config["WEBHOOK_URL"] = WEBHOOK_URL
app.config["SMS_GATE_API_URL"] = f"http://{SMSGATE_PHONE_IP}:8080" if SMSGATE_PHONE_IP else None

print("Loading model and processing data...")
model = xgb.XGBRegressor()
model.load_model("market_timing_v2.json")
market_data = process_market_trends("market_prices.csv")


def verify_signature(raw_body, timestamp, signature, secret_key):
    if not timestamp or not signature:
        return False
    
    data_to_sign = raw_body + timestamp.encode('utf-8')

    expected_hash = hmac.new(
        key=secret_key.encode('utf-8'),
        msg=data_to_sign,
        digestmod=hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(expected_hash, signature)

def register_webhook() -> None | str:
    if not all(
        [
            current_app.config.get("SMS_GATE_API_USERNAME"),
            current_app.config.get("SMS_GATE_API_PASSWORD"),
            current_app.config.get("WEBHOOK_URL"),
        ]
    ):
        return

    with httpx.Client() as client:
        auth = (
            current_app.config["SMS_GATE_API_USERNAME"],
            current_app.config["SMS_GATE_API_PASSWORD"],
        )
        
        # Clean up existing orphaned webhooks
        try:
            get_response = client.get(f"{current_app.config['SMS_GATE_API_URL']}/webhooks", auth=auth)
            if get_response.status_code == 200:
                for wh in get_response.json():
                    if wh.get("url") == current_app.config["WEBHOOK_URL"]:
                        client.delete(f"{current_app.config['SMS_GATE_API_URL']}/webhooks/{wh['id']}", auth=auth)
                        print(f"Cleaned up orphaned webhook ID: {wh['id']}")
        except Exception as e:
            print(f"Warning: Failed to clean up orphaned webhooks: {e}")

        response = client.post(
            f"{current_app.config['SMS_GATE_API_URL']}/webhooks",
            auth=auth,
            json={"url": current_app.config["WEBHOOK_URL"], "event": "sms:received"},
        )
        response.raise_for_status()
        webhook_id = response.json()["id"]
        current_app.config["WEBHOOK_ID"] = webhook_id

        print(f"Registered webhook ID: {webhook_id}")

        return webhook_id


def unregister_webhook(webhook_id: str | None):
    if not all(
        [
            current_app.config.get("SMS_GATE_API_USERNAME"),
            current_app.config.get("SMS_GATE_API_PASSWORD"),
            current_app.config.get("WEBHOOK_ID"),
        ]
    ):
        return

    with httpx.Client() as client:
        response = client.delete(
            f"{current_app.config['SMS_GATE_API_URL']}/webhooks/{current_app.config["WEBHOOK_ID"]}",
            auth=(
                current_app.config["SMS_GATE_API_USERNAME"],
                current_app.config["SMS_GATE_API_PASSWORD"],
            ),
        )
        response.raise_for_status()
        print(f"Unregistered webhook ID: {current_app.config["WEBHOOK_ID"]}")


def send_sms_reply(phone_number, message_text):
    if not phone_number or not message_text:
        print("Error: Missing required query parameters: to, message")
        return

    target_url = f"http://{SMSGATE_PHONE_IP}:8080/messages"
    headers = {
        'Content-Type': 'application/json',
    }
    payload = {
        'message': message_text,
        'phoneNumbers': [phone_number],
    }

    try:
        auth = HTTPBasicAuth(SMSGATE_API_USERNAME, SMSGATE_API_PASSWORD)

        response = requests.post(target_url, json=payload, headers=headers, auth=auth, timeout=10)
        
        if response.status_code not in [200, 201, 202]:
            print(f"SMS Gateway returned error {response.status_code}: {response.text}")
        else:
            print(f"SMS sent successfully (Status {response.status_code})")
    
    except requests.exceptions.RequestException as e:
        print(f"SMS Gateway error: {str(e)}")


@app.route('/webhook/sms-received', methods=['POST'])
def receive_sms():
    raw_body = request.get_data()
    timestamp = request.headers.get('X-Timestamp')
    signature = request.headers.get('X-Signature')

    if not verify_signature(raw_body, timestamp, signature, SIGNING_KEY):
        return jsonify(error="Invalid signature"), 401
    
    try:
        data = json.loads(raw_body)
    except json.JSONDecodeError:
        return jsonify({"error": "Malformed JSON payload"}), 400
    
    event_type = data.get('event')
    message_id = data.get('payload', {}).get('id', 'unknown')
    
    print(f"RECEIVED WEBHOOK: Event={event_type} | MsgID={message_id}")

    if event_type == 'sms:received':
        payload = data.get('payload', {})
        phone_number = payload.get('sender')
        message_text = payload.get('message', '').strip()
        
        def process_request():
            try:
                print(f"Processing MsgID={message_id} in background...")
                parts = [p.strip() for p in message_text.split(',')]
                if len(parts) >= 4:
                    requested_commodity = parts[0]
                    current_season = parts[1]
                    current_weather = parts[2]
                    
                    try:
                        days_held = int(parts[3])
                    except ValueError:
                        send_sms_reply(phone_number, f"Value error: '{parts[3]}' is not a valid number for days held.")
                        return
                    
                    response_message = handle_farmer_sms(
                        requested_commodity=requested_commodity, 
                        current_season=current_season, 
                        current_weather=current_weather, 
                        days_held=days_held,
                        model=model, 
                        results_df=market_data
                    )
                    send_sms_reply(phone_number, response_message)
                else:
                    send_sms_reply(phone_number, "Format error. Format: crop name, season (dry or wet), " \
                    "weather (sunny, normal, drought, flood), # of days held. Example: tomato, dry, sunny, 30")
            except Exception as e:
                print(f"Background processing error: {e}")
                send_sms_reply(phone_number, "An unexpected error occurred while processing your request.")

        # Process in background to avoid webhook timeout
        threading.Thread(target=process_request).start()
        
        return jsonify({"status": "accepted"}), 202
    
    else:
        return jsonify({"error": "Unsupported event type"}), 400


if __name__ == '__main__':
    with app.app_context():
        webhook_id = register_webhook()
    
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        with app.app_context():
            unregister_webhook(webhook_id)
