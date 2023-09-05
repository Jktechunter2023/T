# import the required libraries
from twilio.rest import Client

# Twilio account details
account_sid = 'ACfe817d3c54b17f574510d0f378903547'
auth_token = 'f8ded4029057a2e427403932fc3ba597'
client = Client(account_sid, auth_token)


def send_sms(receiver_no, content):
    # details of the message to be sent
    message = client.messages.create(
        to=receiver_no,# '+33766121245', # replace with your mobile number
        from_='+13184952692', # replace with your Twilio phone number
        body= content) #'Alert from Copernilab')
    # print the message ID
    print('Message ID:', message.sid)
    return message.sid
