document.addEventListener('DOMContentLoaded', function() {
    // Slideshow (slick.js)
    $('.slider').slick({
        arrows: false,
        dots: true,
        appendDots: '.slider-dots',
        dotsClass: 'dots'
    });

    // Hamburger Menu
    let hamberger = document.querySelector('.hamberger');
    let times = document.querySelector('.times');
    let mobileNav = document.querySelector('.mobile-nav');

    if (hamberger && times && mobileNav) {
        hamberger.addEventListener('click', function() {
            mobileNav.classList.add('open');
        });

        times.addEventListener('click', function() {
            mobileNav.classList.remove('open');
        });
    }

    // Chatbot Modal
    const chatbotBtn = document.getElementById('chatbot-btn');
    const chatbotModal = document.getElementById('chatbot-modal');
    const closeBtn = document.getElementById('close-btn');
    const sendBtn = document.getElementById('send-btn');
    const userInput = document.getElementById('user-input');
    const chatbotMessages = document.getElementById('chatbot-messages');

    // Open chatbot modal
    if (chatbotBtn && chatbotModal && closeBtn) {
        chatbotBtn.addEventListener('click', function() {
            chatbotModal.style.display = 'block'; // Show the modal
        });

        closeBtn.addEventListener('click', function() {
            chatbotModal.style.display = 'none'; // Hide the modal
        });
    }

    // Send message logic
    if (sendBtn && userInput && chatbotMessages) {
        sendBtn.addEventListener('click', function() {
            const userText = userInput.value.trim();
            if (userText) {
                appendMessage('You: ' + userText);
                getBotResponse(userText);
                userInput.value = ''; // Clear input field
            }
        });

        // Function to append messages to the chat window
        function appendMessage(message) {
            const p = document.createElement('p');
            p.textContent = message;
            chatbotMessages.appendChild(p);
            chatbotMessages.scrollTop = chatbotMessages.scrollHeight; // Scroll to bottom
        }

        // Function to get bot responses (predefined answers)
        function getBotResponse(userText) {
            const responses = {
                "hello": "Hi there! How can I help you today?",
                "how are you": "I'm doing great, thank you!",
                "what is AI": "AI is Artificial Intelligence, which enables machines to think and learn like humans.",
                "who are you": "I am your virtual assistant, powered by AI to assist you with any questions you may have.",
                "what is cybersecurity": "Cybersecurity refers to the practice of protecting systems, networks, and data from digital attacks, damage, or unauthorized access.",
                "tell me about AI Tech Solutions": "AI Tech Solutions is a start-up focused on providing AI-driven solutions to enhance digital experiences and streamline business operations.",
                "how can AI improve my business": "AI can improve your business by automating processes, enhancing decision-making, increasing efficiency, and providing valuable insights from data.",
                "what services do you offer": "We offer AI-powered networking, cybersecurity, server infrastructure, and IT consulting to help businesses optimize their technology landscape.",
                "what is machine learning": "Machine learning is a subset of AI that enables systems to learn from data and improve their performance over time without being explicitly programmed.",
                "what is deep learning": "Deep learning is a type of machine learning that uses neural networks with many layers to analyze complex patterns and make predictions.",
                "how secure is AI": "AI security involves implementing measures to protect AI systems from attacks and ensuring that AI applications function safely and ethically.",
                "can AI help with customer service": "Yes, AI can automate customer service tasks, provide virtual assistants, and handle customer inquiries to improve response time and customer satisfaction.",
                "hi":"hello, nice to meet you",
                "default": "Sorry, I didn't understand that. Can you please rephrase?"
            };


            const response = responses[userText.toLowerCase()] || responses["default"];
            setTimeout(function() {
                appendMessage('Bot: ' + response);
            }, 1000);
        }
    }

    document.addEventListener('DOMContentLoaded', function() {
        const loginButton = document.querySelector('.admin-login-btn');
        if (loginButton) {
            loginButton.addEventListener('click', function() {
                window.location.href = 'login.html';  // Redirect to login page
            });
        }
    });



    // Feedback Form validation
    document.getElementById("feedbackForm").addEventListener("submit", function(event) {
        const rating = document.querySelector('input[name="rating"]:checked');
        const comment = document.getElementById("comment");

        // Check if rating is selected
        if (!rating) {
            alert("Please provide a rating.");
            event.preventDefault(); // Prevent form submission
            return;
        }

        // Check if comment is provided
        if (comment.value.trim() === "") {
            alert("Please provide your comments.");
            event.preventDefault(); // Prevent form submission
            return;
        }
    });
  });



// for lawai
document.getElementById('queryForm').onsubmit = function(event) {
    event.preventDefault();

    const queryText = document.getElementById('queryInput').value;
    const responseDiv = document.getElementById('response');
    const recommendationsDiv = document.getElementById('recommendations');
    const processingDiv = document.getElementById('processing');

    responseDiv.innerHTML = '';
    recommendationsDiv.innerHTML = '';
    processingDiv.style.display = 'block'; // Show processing message

    fetch('/query', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: queryText }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            responseDiv.innerHTML = `Error: ${data.error}`;
        } else {
            responseDiv.innerHTML = `
                <h2>Relevant Incident:</h2>
                <p><strong>Title:</strong> ${data.relevant_incident.title}</p>
                <p><strong>Description:</strong> ${data.relevant_incident.description}</p>
            `;

            recommendationsDiv.innerHTML = `
                <h2>Recommendations:</h2>
                <p>${data.matched_crime_type}</p>
                <p><strong>BERT Recommendations:</strong><br>${data.bert_recommendations}</p>
                <p><strong>BART Recommendations:</strong><br>${data.bart_recommendations}</p>
            `;
        }
        processingDiv.style.display = 'none'; // Hide processing message
    })
    .catch(error => {
        responseDiv.innerHTML = `Error: ${error.message}`;
        processingDiv.style.display = 'none'; // Hide processing message
    });
}


//Section for the customer inquiry form
const scriptURL = 'https://script.google.com/macros/s/AKfycbxhLZFDopLT_sbbegXhQvRCTYPxDJKINHLfv7oI2Slw1DEGLf244B-jr57LDwyzJNdw/exec'; // Replace with your Google Apps Script URL
const form = document.forms['customer-inquiries-form'];
const msg = document.getElementById("msg");

form.addEventListener('submit', (e) => {
    e.preventDefault();
    fetch(scriptURL, { method: 'POST', body: new FormData(form) })
        .then((response) => {
            msg.innerHTML = "Message sent successfully";
            setTimeout(() => {
                msg.innerHTML = "";
            }, 5000);
            form.reset();
        })
        .catch((error) => {
            console.error('Error!', error.message);
            msg.innerHTML = "Error in sending message. Please try again.";
            setTimeout(() => {
                msg.innerHTML = "";
            }, 5000);
        });
});
