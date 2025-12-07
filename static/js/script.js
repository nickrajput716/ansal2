const landingPage = document.getElementById('landingPage');
const predictionPage = document.getElementById('predictionPage');
const resultsPage = document.getElementById('resultsPage');
const getStartedBtn = document.getElementById('getStartedBtn');
const backBtn = document.getElementById('backBtn');
const backFromResultsBtn = document.getElementById('backFromResultsBtn');
const predictionForm = document.getElementById('predictionForm');
const tryAgainBtn = document.getElementById('tryAgainBtn');
const loadingOverlay = document.getElementById('loadingOverlay');
const errorMessage = document.getElementById('errorMessage');

getStartedBtn.addEventListener('click', () => {
    landingPage.style.display = 'none';
    predictionPage.style.display = 'block';
    window.scrollTo(0, 0);
});

backBtn.addEventListener('click', () => {
    predictionPage.style.display = 'none';
    landingPage.style.display = 'block';
    window.scrollTo(0, 0);
});

backFromResultsBtn.addEventListener('click', () => {
    resultsPage.style.display = 'none';
    predictionPage.style.display = 'block';
    window.scrollTo(0, 0);
});

tryAgainBtn.addEventListener('click', () => {
    resultsPage.style.display = 'none';
    predictionPage.style.display = 'block';
    predictionForm.reset();
    window.scrollTo(0, 0);
});

predictionForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    errorMessage.style.display = 'none';

    const formData = new FormData(predictionForm);
    const data = Object.fromEntries(formData);

    for (let key in data) {
        if (data[key] === '') {
            errorMessage.textContent = 'Please fill in all fields';
            errorMessage.style.display = 'block';
            return;
        }
    }

    Object.keys(data).forEach(key => {
        if (key.includes('Hours') || key.includes('Scores') || key.includes('Attendance') ||
            key.includes('Sleep') || key.includes('Activity') || key.includes('Sessions')) {
            data[key] = parseFloat(data[key]);
        }
    });

    loadingOverlay.style.display = 'flex';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        if (!response.ok) {
            throw new Error('Prediction failed');
        }

        const result = await response.json();
        displayResults(result);
    } catch (err) {
        errorMessage.textContent = 'Error: ' + (err.message || 'An error occurred');
        errorMessage.style.display = 'block';
        loadingOverlay.style.display = 'none';
        window.scrollTo(0, 0);
    }
});

function displayResults(data) {
    loadingOverlay.style.display = 'none';
    predictionPage.style.display = 'none';
    resultsPage.style.display = 'block';
    window.scrollTo(0, 0);

    animateValue('scoreValue', 0, data.prediction, 1500);
    document.getElementById('gradeDisplay').textContent = data.grade;
    document.getElementById('categoryLabel').textContent = data.category;
    document.getElementById('percentileValue').textContent = data.percentile.toFixed(1);
    document.getElementById('r2Score').textContent = data.metrics.r2_score.toFixed(3);
    document.getElementById('maeScore').textContent = data.metrics.mae.toFixed(2);

    const r2 = data.metrics.r2_score;
    let quality = 'Good';
    if (r2 >= 0.9) quality = 'Excellent';
    else if (r2 >= 0.8) quality = 'Very Good';
    else if (r2 >= 0.7) quality = 'Good';
    document.getElementById('qualityScore').textContent = quality;

    const scoreCircle = document.querySelector('.score-circle');
    if (data.grade === 'A+' || data.grade === 'A') {
        scoreCircle.style.borderColor = 'rgba(16, 185, 129, 0.8)';
    } else if (data.grade === 'B') {
        scoreCircle.style.borderColor = 'rgba(102, 126, 234, 0.8)';
    } else if (data.grade === 'C') {
        scoreCircle.style.borderColor = 'rgba(245, 158, 11, 0.8)';
    } else {
        scoreCircle.style.borderColor = 'rgba(239, 68, 68, 0.8)';
    }

    displayInsights(data.insights);
    displayRecommendations(data.recommendations);
    displayGraphs(data.graphs);
}

function displayInsights(insights) {
    const container = document.getElementById('insightsGrid');
    container.innerHTML = '';

    insights.forEach((insight, index) => {
        const div = document.createElement('div');
        div.className = `insight-item ${insight.type} slide-up`;
        div.style.animationDelay = `${index * 0.1}s`;
        div.innerHTML = `
            <h4><span style="font-size: 1.3rem;">${insight.icon}</span> ${insight.title}</h4>
            <p>${insight.text}</p>
        `;
        container.appendChild(div);
    });
}

function displayRecommendations(recommendations) {
    const container = document.getElementById('recommendationsGrid');
    container.innerHTML = '';

    if (recommendations.length === 0) {
        container.innerHTML = '<p style="color: #cbd5e1; text-align: center; grid-column: 1 / -1;">Great! You\'re already following best practices. Keep it up!</p>';
        return;
    }

    recommendations.forEach((rec, index) => {
        const div = document.createElement('div');
        div.className = 'recommendation-item slide-up';
        div.style.animationDelay = `${index * 0.1}s`;

        const priorityClass = `priority-${rec.priority.toLowerCase()}`;

        div.innerHTML = `
            <h4>💡 ${rec.title}</h4>
            <div class="action">📌 ${rec.action}</div>
            <div class="impact">✅ Impact: ${rec.impact}</div>
            <span class="priority-badge ${priorityClass}">Priority: ${rec.priority}</span>
        `;
        container.appendChild(div);
    });
}

function displayGraphs(graphs) {
    const container = document.getElementById('graphsContainer');
    container.innerHTML = '';

    const graphTitles = {
        'distribution': '📊 Your Score vs All Students',
        'study_hours': '📚 Study Hours Impact on Performance',
        'attendance': '✅ Attendance vs Performance Analysis',
        'sleep': '😴 Sleep Hours Effect on Scores',
        'importance': '🎯 Top Factors Affecting Your Performance',
        'previous_score': '📈 Previous Score vs Predicted Score'
    };

    let graphIndex = 0;
    for (const [key, imgData] of Object.entries(graphs)) {
        const div = document.createElement('div');
        div.className = 'graph-item';
        div.style.animationDelay = `${graphIndex * 0.1}s`;
        
        const img = document.createElement('img');
        img.src = `data:image/png;base64,${imgData}`;
        img.alt = graphTitles[key] || key;
        
        // Make graph clickable - opens in new tab
        div.addEventListener('click', () => {
            openGraphInNewTab(imgData, graphTitles[key] || key);
        });
        
        // Add keyboard accessibility
        div.setAttribute('tabindex', '0');
        div.setAttribute('role', 'button');
        div.setAttribute('aria-label', `Click to enlarge ${graphTitles[key] || key}`);
        
        div.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                openGraphInNewTab(imgData, graphTitles[key] || key);
            }
        });
        
        div.appendChild(img);
        container.appendChild(div);
        graphIndex++;
    }
}

function openGraphInNewTab(base64Data, title) {
    // Create a new window with the graph
    const newWindow = window.open('', '_blank');
    
    if (newWindow) {
        newWindow.document.write(`
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>${title}</title>
                <style>
                    * {
                        margin: 0;
                        padding: 0;
                        box-sizing: border-box;
                    }
                    body {
                        background: #0f0f23;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        min-height: 100vh;
                        padding: 2rem;
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    }
                    .header {
                        background: rgba(26, 26, 46, 0.8);
                        backdrop-filter: blur(20px);
                        border: 1px solid rgba(255, 255, 255, 0.1);
                        border-radius: 1rem;
                        padding: 1.5rem 2rem;
                        margin-bottom: 2rem;
                        text-align: center;
                    }
                    h1 {
                        color: white;
                        font-size: 1.5rem;
                        margin-bottom: 0.5rem;
                    }
                    .subtitle {
                        color: #94a3b8;
                        font-size: 0.9rem;
                    }
                    .graph-container {
                        background: rgba(26, 26, 46, 0.8);
                        backdrop-filter: blur(20px);
                        border: 1px solid rgba(255, 255, 255, 0.1);
                        border-radius: 1.5rem;
                        padding: 2rem;
                        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
                        max-width: 1200px;
                        width: 100%;
                    }
                    img {
                        width: 100%;
                        height: auto;
                        border-radius: 0.5rem;
                        display: block;
                    }
                    .actions {
                        margin-top: 1.5rem;
                        display: flex;
                        gap: 1rem;
                        justify-content: center;
                        flex-wrap: wrap;
                    }
                    button {
                        padding: 0.75rem 1.5rem;
                        border: none;
                        border-radius: 0.5rem;
                        font-size: 0.9rem;
                        font-weight: 600;
                        cursor: pointer;
                        transition: all 0.3s ease;
                    }
                    .btn-download {
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                    }
                    .btn-download:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
                    }
                    .btn-close {
                        background: rgba(75, 85, 99, 0.5);
                        color: #cbd5e1;
                        border: 1px solid rgba(255, 255, 255, 0.2);
                    }
                    .btn-close:hover {
                        background: rgba(75, 85, 99, 0.7);
                        color: white;
                    }
                    @media (max-width: 768px) {
                        body {
                            padding: 1rem;
                        }
                        .graph-container {
                            padding: 1rem;
                        }
                        h1 {
                            font-size: 1.2rem;
                        }
                    }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>${title}</h1>
                    <p class="subtitle">Student Performance Analysis</p>
                </div>
                <div class="graph-container">
                    <img src="data:image/png;base64,${base64Data}" alt="${title}">
                    <div class="actions">
                        <button class="btn-download" onclick="downloadGraph()">
                            📥 Download Graph
                        </button>
                        <button class="btn-close" onclick="window.close()">
                            ✕ Close
                        </button>
                    </div>
                </div>
                <script>
                    function downloadGraph() {
                        const link = document.createElement('a');
                        link.href = 'data:image/png;base64,${base64Data}';
                        link.download = '${title.replace(/[^a-z0-9]/gi, '_').toLowerCase()}.png';
                        link.click();
                    }
                </script>
            </body>
            </html>
        `);
        newWindow.document.close();
    } else {
        // Fallback if popup is blocked
        alert('Please allow popups to view the graph in full size. Alternatively, right-click the graph and select "Open image in new tab".');
    }
}

function animateValue(id, start, end, duration) {
    const element = document.getElementById(id);
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;

    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            current = end;
            clearInterval(timer);
        }
        element.textContent = current.toFixed(1);
    }, 16);
}

// Add smooth scroll behavior for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});