// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    try {
        const res = await fetch('/api/crops');
        const data = await res.json();
        const select = document.getElementById('cropSelect');
        data.crops.forEach(crop => {
            const option = document.createElement('option');
            option.value = crop;
            option.textContent = crop;
            select.appendChild(option);
        });
    } catch (e) {
        console.error("Failed to load crops:", e);
    }
});

// Update range values
function updateVal(id) {
    document.getElementById(`${id}-val`).innerText = document.getElementById(id).value;
}

// Tab Switching
function switchTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    
    document.getElementById(tabId).classList.add('active');
    event.currentTarget.classList.add('active');
}

// Predict Yield
async function predictYield() {
    const btn = event.currentTarget;
    const origText = btn.innerText;
    btn.innerText = "⏳ Predicting...";
    btn.disabled = true;

    const reqData = {
        N: parseFloat(document.getElementById('N').value),
        P: parseFloat(document.getElementById('P').value),
        K: parseFloat(document.getElementById('K').value),
        pH: parseFloat(document.getElementById('pH').value),
        temp: parseFloat(document.getElementById('temp').value),
        rainfall: parseFloat(document.getElementById('rain').value),
        humidity: parseFloat(document.getElementById('hum').value),
        solar: parseFloat(document.getElementById('solar').value),
        crop: document.getElementById('cropSelect').value,
        area: parseFloat(document.getElementById('area').value),
        pesticide: parseFloat(document.getElementById('pest').value),
        season: document.getElementById('seasonSelect').value,
    };

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(reqData)
        });
        
        const data = await res.json();
        
        document.getElementById('yield-output').innerHTML = `${data.yield} <small style="font-size: 1.5rem">t/ha</small>`;
        document.getElementById('yield-cat').innerText = data.category;
        document.getElementById('prod-output').innerText = `${data.production.toLocaleString()} tonnes`;
        
        document.getElementById('predict-result').classList.remove('hidden');
    } catch (e) {
        alert("An error occurred while predicting.");
        console.error(e);
    } finally {
        btn.innerText = origText;
        btn.disabled = false;
    }
}

// Recommend Crops
async function recommendCrops() {
    const btn = event.currentTarget;
    const origText = btn.innerText;
    btn.innerText = "⏳ Analyzing...";
    btn.disabled = true;

    const reqData = {
        N: parseFloat(document.getElementById('rN').value),
        P: parseFloat(document.getElementById('rP').value),
        K: parseFloat(document.getElementById('rK').value),
        pH: parseFloat(document.getElementById('rpH').value),
        temp: parseFloat(document.getElementById('rtemp').value),
        rainfall: parseFloat(document.getElementById('rrain').value),
        humidity: parseFloat(document.getElementById('rhum').value)
    };

    try {
        const res = await fetch('/api/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(reqData)
        });
        
        const data = await res.json();
        const grid = document.getElementById('rec-results');
        grid.innerHTML = '';
        
        const medals = ["🥇", "🥈", "🥉"];
        
        data.recommendations.forEach((rec, i) => {
            const card = document.createElement('div');
            card.className = 'rec-card';
            card.innerHTML = `
                <div style="font-size:3rem">${medals[i] || ''}</div>
                <h2>${rec.crop}</h2>
                <div class="score">${rec.suitability}</div>
                <div style="font-size: 0.9rem; color: #6c757d; text-transform: uppercase;">Suitability Score</div>
            `;
            grid.appendChild(card);
        });
        
        grid.classList.remove('hidden');
    } catch (e) {
        alert("An error occurred while getting recommendations.");
        console.error(e);
    } finally {
        btn.innerText = origText;
        btn.disabled = false;
    }
}
