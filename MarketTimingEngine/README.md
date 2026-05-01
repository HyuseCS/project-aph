# Market Timing Webhook Server

## Setup Instructions

1. **Connect Devices**: Connect your Linux PC and Android phone via Tailscale.
2. **Install Dependencies**: Ensure you are in a virtual environment (optional but recommended), then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. **Expose the Webhook**: Get your public URL by running Tailscale:
   ```bash
   tailscale serve --bg 5000
   ```
   *Note: MAKE SURE MAGICDNS AND HTTPS CERTIFICATE ARE ENABLED IN THE TAILSCALE ADMIN CONSOLE IN THE BROWSER.*

## Environment Configuration

Create a `.env` file in the same directory with the following contents:

```env
SMSGATE_PHONE_IP="your_phone_ip"          # In SMSGate app, enable local server; see Local address
SMSGATE_API_USERNAME="username"           # SMSGate Local server - username
SMSGATE_API_PASSWORD="password"           # SMSGate Local server - password
WEBHOOK_URL="https://<url_here>/webhook/sms-received" # URL from tailscale serve command
SIGNING_KEY="temp"                        # Temporary placeholder
```

## Running and Bootstrapping

The application automatically registers its webhook with the SMSGate app on startup and unregisters it on shutdown. However, you need the `SIGNING_KEY` to securely verify incoming SMS messages, which is generated *after* the webhook is registered.

1. **Start the server** for the first time to register the webhook:
   ```bash
   python webhook_server.py
   ```
2. **Get the Signing Key**: On your phone, go to **SMSGate settings > Webhooks**. Look at the newly registered webhook and copy the generated **Signing Key**.
3. **Update `.env`**: Stop the server (Ctrl+C), update `SIGNING_KEY` in your `.env` file with the actual key from the app, and **restart the server**.

The server is now fully configured and ready to process SMS requests dynamically!
