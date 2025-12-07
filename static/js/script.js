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
        scoreCircle.style.borderColor = 'rgba(6, 182, 212, 0.8)';
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
        div.innerHTML = `<img src="data:image/png;base64,${imgData}" alt="${graphTitles[key] || key}">`;
        container.appendChild(div);
        graphIndex++;
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
