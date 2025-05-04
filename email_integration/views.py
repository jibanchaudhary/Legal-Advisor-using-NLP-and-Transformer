from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from .gmail import *
import speech_recognition as sr


def index(request):
    return render(request, "index.html")


def record_page(request):
    return render(request, "record.html")


def record_audio(request):
    STOP_COMMANDS = ["stop", "रोक"]
    try:
        if request.method == "POST":
            nepali_texts = []
            english_texts = []
            while True:
                nepali_text = record_text()
                if nepali_text:
                    if nepali_text.lower() in STOP_COMMANDS:
                        print("Exit command detected. Exiting ....")
                        return JsonResponse({"message": "Stopped recording"})

                    nepali_texts.append(nepali_text)
                else:
                    break

            for nepali_text in nepali_texts:
                english_text = nepali_trans(nepali_text)
                english_texts.append(english_text)

            return JsonResponse(
                {
                    "nepali_text": nepali_texts,
                    "english_text": english_texts,
                }
            )
    except Exception as e:
        return JsonResponse({"error": f"Error occurred: {str(e)}"})
    return JsonResponse({"error": "Invalid request method"})


# Email sending
def send_transcribed_email(request):
    if request.method == "POST":
        raw_content = request.POST.get("transcription")

        fields = extracts_template_field(raw_content)
        summary = generate_email_summary(raw_content)
        edited_text = bart_edit(summary)
        email_content = report_template(
            fields["subject"], fields["reporter_name"], edited_text
        )

        if "preview" in request.POST:
            return JsonResponse(
                {
                    "email_content": email_content,
                    "subject": fields["subject"],
                    "reporter_name": fields["reporter_name"],
                }
            )

        recipient_email = "recipient@example.com"
        response = send_email(
            email_content, fields["subject"], recipient_email, fields["reporter_name"]
        )
        return JsonResponse(
            {"status": "Email sent successfully!", "response": response}
        )

    return JsonResponse({"error": "Invalid request method"})
