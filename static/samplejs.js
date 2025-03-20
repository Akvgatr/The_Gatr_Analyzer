// Change Background Color Functionality
document.getElementById('colorButton').addEventListener('click', function() {
    const colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#FFB833'];
    const randomColor = colors[Math.floor(Math.random() * colors.length)];
    document.body.style.backgroundColor = randomColor;
});

// Form Submission Handling
document.getElementById('contactForm').addEventListener('submit', function(event) {
    event.preventDefault();
    const name = document.getElementById('name').value;
    const email = document.getElementById('email').value;
    const message = document.getElementById('message').value;

    if (name && email && message) {
        document.getElementById('formStatus').textContent = "Thank you for your message, " + name + "!";
        document.getElementById('formStatus').style.color = "green";
        this.reset(); // Clear the form
    } else {
        document.getElementById('formStatus').textContent = "Please fill in all fields.";
        document.getElementById('formStatus').style.color = "red";
    }
});

// Simple Animation for Form Status
function animateStatus() {
    const statusElement = document.getElementById('formStatus');
    statusElement.style.transition = "transform 0.5s ease";
    statusElement.style.transform = "scale(1.2)";
    setTimeout(() => {
        statusElement.style.transform = "scale(1)";
    }, 500);
}

document.getElementById('contactForm').addEventListener('submit', animateStatus);

// Additional Interactions
document.querySelectorAll('nav ul li a').forEach(link => {
    link.addEventListener('click', function(event) {
        event.preventDefault();
        const targetId = this.getAttribute('href').substring(1);
        const targetSection = document.getElementById(targetId);
        window.scrollTo({
            top: targetSection.offsetTop,
            behavior: 'smooth'
        });
    });
});
