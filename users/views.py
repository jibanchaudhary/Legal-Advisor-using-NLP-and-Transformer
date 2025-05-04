from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from .models import Feedback

def submit_feedback(request):
    if request.method == "POST":
        name = request.POST.get("name")
        email = request.POST.get("email")
        phone = request.POST.get("phone")
        company = request.POST.get("company")
        country = request.POST.get("country")
        job_title = request.POST.get("job_title")
        queries = request.POST.get("queries")  # Make sure this matches the form field name

        Feedback.objects.create(
            name=name,
            email=email,
            phone=phone,
            company=company,
            country=country,
            job_title=job_title,
            queries=queries,  
        )

        return redirect("index")  

    return render(request, "index.html")


def user_login(request):
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('index')  # Change this to your desired page
    else:
        form = AuthenticationForm()
    return render(request, 'users/login.html', {'form': form})

# Logout View
def user_logout(request):
    logout(request)
    return redirect('index')

# Register View
def user_register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('users:user_login')
    else:
        form = UserCreationForm()
    return render(request, 'users/register.html', {'form': form})
