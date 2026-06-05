# Run this first in terminal: pip install boto3
import sys
import boto3
import json

# AWS Credentials

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

MODEL = "anthropic.claude-3-haiku-20240307-v1:0"
history = []

print("Streaming CLI Chatbot")
print("Type 'exit' to quit.\n")

while True:
    sys.stdout.write("User: ")
    sys.stdout.flush()
    user_input = sys.stdin.readline().strip()

    if user_input.lower() == "exit":
        print("Exiting chat. Chat Ended")
        break

    history.append({"role": "user", "content": [{"text": user_input}]})

    if len(history) > 20:
        history = history[-20:]

    # print("Assistant:", end=" ", flush=True)
    sys.stdout.write("Assistant: ")
    sys.stdout.flush()

    response = bedrock_runtime.converse_stream(
        modelId=MODEL,
        messages=history,
        inferenceConfig={
            "maxTokens": 100,
            "temperature": 0.3,
            "topP": 0.7
        }
    )

    assistant_reply = ""

    for event in response["stream"]:
        if "contentBlockDelta" in event:
            text = event["contentBlockDelta"]["delta"].get("text", "")
            sys.stdout.write(text)
            sys.stdout.flush()
            assistant_reply += text

    print("\n")  # ✅ FIXED: moved inside the while loop (was outside/misindented)

    history.append({"role": "assistant", "content": [{"text": assistant_reply}]})
    # ✅ FIXED: moved inside the while loop (was outside, so reply was never saved mid-chat)


    ####Your bedrock_dev user doesn't have IAM admin permissions — it can't modify its own policies. You need to do this via the AWS Console with the root/admin account.
# Do this in AWS Console:

# Go to console.aws.amazon.com and login with your root account (the email you used to create AWS account)
# Navigate to IAM → Users → bedrock_dev
# Click Add permissions → Attach policies directly
# Search and enable these two:

# ✅ AmazonBedrockFullAccess
# Your keys are fine — the issue is purely that bedrock_dev was created with limited permissions. Only the root/admin account can grant it access.

# Do you have access to the root AWS account login? If yes, the console fix takes less than 2 minutes.



