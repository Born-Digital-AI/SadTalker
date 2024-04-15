import os
import smtplib
from email.mime.text import MIMEText
import logging as log

SMTP_SERVER = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.environ.get('SMTP_PORT', '465'))
SMTP_SENDER = os.environ.get('SMTP_SENDER', 'bdavatargen@gmail.com')
SMTP_PWD = os.environ.get('SMTP_PWD')


class EmailSender:
    def __init__(self) -> None:
        self.sender = SMTP_SENDER
        self.password = SMTP_PWD

    def send(self, subject, body, recipients):
        msg = MIMEText(body, 'html')
        msg['Subject'] = subject
        msg['From'] = self.sender
        msg['To'] = ', '.join(recipients)

        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as smtp_server:
            smtp_server.login(self.sender, self.password)
            smtp_server.sendmail(self.sender, recipients, msg.as_string())
        log.info(f"Message sent to email address: {msg['To']}")
