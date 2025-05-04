// var tabLinks = document.getElementsByClassName("tab-links");
// var tabContents = document.getElementsByClassName("tab-contents");

// function openTab(event, tabName) {
//     for (let tabLink of tabLinks) {
//         tabLink.classList.remove("active-link");
//     }
//     for (let tabContent of tabContents) {
//         tabContent.classList.remove("active-tab");
//     }

//     event.currentTarget.classList.add("active-link");
//     document.getElementById(tabName).classList.add("active-tab");
// }

// const scriptURL = 'https://script.google.com/macros/s/AKfycbx0nHoLlCDBF2LVKYzZHf3kPknc0p_kQIbEirKimrmDv7vSYJ1XPDI7kdT9QNc0UipI/exec'; // Replace with your Google Apps Script URL
// const form = document.forms['customer-inquiries-form'];
// const msg = document.getElementById("msg");

// form.addEventListener('submit', (e) => {
//     e.preventDefault();
//     fetch(scriptURL, { method: 'POST', body: new FormData(form) })
//         .then((response) => {
//             msg.innerHTML = "Message sent successfully";
//             setTimeout(() => {
//                 msg.innerHTML = "";
//             }, 5000);
//             form.reset();
//         })
//         .catch((error) => {
//             console.error('Error!', error.message);
//             msg.innerHTML = "Error in sending message. Please try again.";
//             setTimeout(() => {
//                 msg.innerHTML = "";
//             }, 5000);
//         });
// });
