# auth.py
import random
import string
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from passlib.context import CryptContext
from dotenv import load_dotenv

load_dotenv()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def generate_otp() -> str:
    return ''.join(random.choices(string.digits, k=6))


def get_otp_expiry() -> str:
    expiry = datetime.now() + timedelta(minutes=10)
    return expiry.strftime("%Y-%m-%d %H:%M:%S")


def is_otp_valid(otp_expires: str) -> bool:
    try:
        expiry = datetime.strptime(otp_expires, "%Y-%m-%d %H:%M:%S")
        return datetime.now() < expiry
    except:
        return False


def send_otp_email(email: str, otp: str, user_name: str) -> bool:
    try:
        sender_email = os.getenv("EMAIL_ADDRESS")
        sender_password = os.getenv("EMAIL_PASSWORD")

        if not sender_email or not sender_password:
            print("⚠️ Email credentials not set. OTP would be:", otp)
            return True

        msg = MIMEMultipart("alternative")
        msg["Subject"] = "🌱 Your CropHealth AI Login OTP"
        msg["From"] = sender_email
        msg["To"] = email

        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <div style="background: #2d6a4f; color: white; padding: 20px; border-radius: 10px 10px 0 0;">
                <h2>🌱 CropHealth AI</h2>
            </div>
            <div style="padding: 30px; background: #f9f9f9;">
                <p>Hello <strong>{user_name}</strong>,</p>
                <p>Your One-Time Password (OTP) for login is:</p>
                <div style="background: #2d6a4f; color: white; font-size: 32px; 
                            letter-spacing: 8px; text-align: center; padding: 20px; 
                            border-radius: 8px; font-weight: bold;">
                    {otp}
                </div>
                <p style="color: #666; margin-top: 20px;">
                    ⏱️ This OTP expires in <strong>10 minutes</strong>.<br>
                    Do not share this with anyone.
                </p>
            </div>
        </body>
        </html>
        """

        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, msg.as_string())

        print(f"✅ OTP sent to {email}")
        return True

    except Exception as e:
        print(f"❌ Email error: {e}")
        print(f"⚠️ OTP for {email} is: {otp}")
        return True  # Still return True for development