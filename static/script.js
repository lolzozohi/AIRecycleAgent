function openTab(event, tabName) {
  // Hide all tab contents
  const tabContents = document.getElementsByClassName('tab-content');
  for (let i = 0; i < tabContents.length; i++) {
      tabContents[i].style.display = 'none';
  }

  // Remove the active class from all tab buttons
  const tabButtons = document.getElementsByClassName('tab-button');
  for (let i = 0; i < tabButtons.length; i++) {
      tabButtons[i].className = tabButtons[i].className.replace(' active', '');
  }

  // Show the current tab and add the active class to the button that opened the tab
  document.getElementById(tabName).style.display = 'block';
  event.currentTarget.className += ' active';
}

// Default open tab
document.addEventListener('DOMContentLoaded', function() {
  document.querySelector('.tab-button').click();
});

function calculateFootprint(type) {
  let totalCarbon = 0;

  if (type === 'simple') {
      const milesDriven = parseFloat(document.getElementById('transportationSimple').value);
      const electricityUsage = parseFloat(document.getElementById('electricitySimple').value);

      const transportationCarbon = milesDriven * 400; // Example: 0.411 lbs CO2 per mile
      const electricityCarbon = electricityUsage * 390.0894; // Example: 0.92 lbs CO2 per kWh

      totalCarbon = transportationCarbon + electricityCarbon;
  } else if (type === 'advanced') {
      const milesDriven = parseFloat(document.getElementById('transportation').value);
      const electricityUsage = parseFloat(document.getElementById('electricity').value);
      const gasUsage = parseFloat(document.getElementById('gas').value);
      const wasteProduction = parseFloat(document.getElementById('waste').value);
      const waterUsage = parseFloat(document.getElementById('water').value);

      const transportationCarbon = milesDriven * 400*4; // Example: 0.411 lbs CO2 per mile
      const electricityCarbon = electricityUsage * 390.0894; // Example: 0.92 lbs CO2 per kWh
      const gasCarbon = gasUsage * 5307.031; // Example: 11.7 lbs CO2 per therm
      const wasteCarbon = wasteProduction * 1018.2*4; // Example: 1018.2 lbs CO2 per lb of waste per week
      const waterCarbon = waterUsage * 0.0022; // Example: 0.0022 lbs CO2 per gallon of water

      totalCarbon = transportationCarbon + electricityCarbon + gasCarbon + wasteCarbon + waterCarbon;
  }

  document.getElementById('carbonResult').textContent = `Carbon Footprint: ${totalCarbon.toFixed(2)} grams CO2 per month.`;
}

function saveData() {
  const transportation = document.getElementById('transportation').value;
  const electricity = document.getElementById('electricity').value;
  const gas = document.getElementById('gas').value;
  const waste = document.getElementById('waste').value;
  const water = document.getElementById('water').value;

  localStorage.setItem('transportation', transportation);
  localStorage.setItem('electricity', electricity);
  localStorage.setItem('gas', gas);
  localStorage.setItem('waste', waste);
  localStorage.setItem('water', water);

  alert('Data saved!');
}

function loadData() {
  const transportation = localStorage.getItem('transportation');
  const electricity = localStorage.getItem('electricity');
  const gas = localStorage.getItem('gas');
  const waste = localStorage.getItem('waste');
  const water = localStorage.getItem('water');

  if (transportation) document.getElementById('transportation').value = transportation;
  if (electricity) document.getElementById('electricity').value = electricity;
  if (gas) document.getElementById('gas').value = gas;
  if (waste) document.getElementById('waste').value = waste;
  if (water) document.getElementById('water').value = water;

  alert('Data loaded!');
}
