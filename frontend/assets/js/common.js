// Common JavaScript for all pages
const API_BASE = 'http://localhost:5000/api';

// Function to show results
function showResult(elementId, message, type = 'success') {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = message;
        element.className = `result ${type}`;
        element.style.display = 'block';
    }
}

// Function to set active navigation
function setActiveNav() {
    const currentPage = window.location.pathname.split('/').pop() || 'index.html';
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        const linkPage = link.getAttribute('href');
        if (linkPage === currentPage) {
            link.classList.add('active');
        }
    });
}

// Function to load components
function loadComponent(componentId, filePath) {
    fetch(filePath)
        .then(response => response.text())
        .then(data => {
            document.getElementById(componentId).innerHTML = data;
            if (componentId === 'header-container') {
                setActiveNav();
            }
        })
        .catch(error => console.error('Error loading component:', error));
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Load header and footer if containers exist
    if (document.getElementById('header-container')) {
        loadComponent('header-container', 'components/header.html');
    }
    if (document.getElementById('footer-container')) {
        loadComponent('footer-container', 'components/footer.html');
    }
    
    setActiveNav();
});

// Common API status check
async function checkAPIStatus() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        const status = await response.json();
        console.log('🌱 CropCareAI Status:', status);
        return status;
    } catch (error) {
        console.log('⚠️ API not reachable, but interface will work');
        return null;
    }
}