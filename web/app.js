const form = document.querySelector('#prediction-form');
const empty = document.querySelector('#empty-state');
const loading = document.querySelector('#loading-state');
const result = document.querySelector('#result');
const error = document.querySelector('#error-state');

const demo = {
  species: 'lion\'s_mane', room_id: 'GR-04', substrate_type: 'supplemented_sawdust', flush_number: '2',
  room_age_days: 17, temperature_c: 23.8, humidity_pct: 95.1, co2_ppm: 1540,
  substrate_moisture_pct: 72, fresh_air_exchanges_hour: 2.1, light_hours: 9,
  previous_yield_kg: 8.4, pin_count_index: 66
};

document.querySelector('#demo-button').addEventListener('click', () => {
  for (const [name, value] of Object.entries(demo)) form.elements[name].value = value;
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  empty.classList.add('hidden'); result.classList.add('hidden'); error.classList.add('hidden'); loading.classList.remove('hidden');
  const button = form.querySelector('button[type="submit"]'); button.disabled = true;
  const data = Object.fromEntries(new FormData(form).entries());
  const numeric = ['room_age_days','temperature_c','humidity_pct','co2_ppm','substrate_moisture_pct','fresh_air_exchanges_hour','light_hours','previous_yield_kg','pin_count_index'];
  numeric.forEach(key => data[key] = Number(data[key]));
  try {
    const response = await fetch('/api/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)});
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || 'Prediction service unavailable');
    document.querySelector('#yield-value').textContent = body.predicted_yield_kg.toFixed(1);
    document.querySelector('#risk-value').textContent = `${Math.round(body.contamination_probability * 100)}%`;
    document.querySelector('#labor-value').textContent = body.recommended_harvest_labor_hours;
    document.querySelector('#crates-value').textContent = body.recommended_crates;
    const badge = document.querySelector('#risk-badge'); badge.textContent = `${body.risk_band} risk`; badge.className = `badge ${body.risk_band}`;
    const list = document.querySelector('#recommendations'); list.innerHTML = '';
    body.recommendations.forEach(item => { const li = document.createElement('li'); li.textContent = item; list.appendChild(li); });
    result.classList.remove('hidden');
  } catch (err) {
    document.querySelector('#error-message').textContent = err.message; error.classList.remove('hidden');
  } finally { loading.classList.add('hidden'); button.disabled = false; }
});

