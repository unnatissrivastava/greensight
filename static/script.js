document.addEventListener("DOMContentLoaded", () => {
  // Live (simulated) sensor readings on the /demo page
  const moistureEl = document.getElementById("sensor-moisture");
  const humidityEl = document.getElementById("sensor-humidity");
  const tempEl = document.getElementById("sensor-temp");

  if (moistureEl && humidityEl && tempEl) {
    let moisture = 32, humidity = 58, temp = 27;

    function tickSensors() {
      moisture = Math.max(10, Math.min(60, moisture + (Math.random() - 0.5) * 3));
      humidity = Math.max(30, Math.min(90, humidity + (Math.random() - 0.5) * 4));
      temp = Math.max(15, Math.min(42, temp + (Math.random() - 0.5) * 1.5));

      moistureEl.textContent = `${moisture.toFixed(1)}%`;
      humidityEl.textContent = `${humidity.toFixed(1)}%`;
      tempEl.textContent = `${temp.toFixed(1)}°C`;
    }

    tickSensors();
    setInterval(tickSensors, 2500);
  }

  // ---------- Scroll reveal (flowy fade-in) ----------
  const revealEls = document.querySelectorAll(".reveal");
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
      } else {
        entry.target.classList.remove("visible");
      }
    });
  }, { threshold: 0.2 });
  revealEls.forEach((el, i) => {
    el.style.transitionDelay = `${(i % 4) * 0.08}s`;
    observer.observe(el);
  });

  // ---------- Floating, swaying trees ----------
  const treeSVG = `
    <svg viewBox="0 0 100 140" xmlns="http://www.w3.org/2000/svg">
      <rect x="46" y="88" width="8" height="52" fill="currentColor"/>
      <circle cx="50" cy="58" r="34" fill="currentColor"/>
      <circle cx="28" cy="74" r="25" fill="currentColor"/>
      <circle cx="72" cy="74" r="25" fill="currentColor"/>
    </svg>`;

  // Fixed "random" layout so it looks intentional and stays consistent on every load
  const layout = [
    { left: '2%',  height: 150, opacity: 0.22, duration: 6.5, delay: 0.0 },
    { left: '13%', height: 100, opacity: 0.14, duration: 5.5, delay: 0.6 },
    { left: '24%', height: 190, opacity: 0.28, duration: 7.2, delay: 1.1 },
    { left: '38%', height: 120, opacity: 0.16, duration: 6.0, delay: 0.3 },
    { left: '52%', height: 170, opacity: 0.24, duration: 6.8, delay: 1.6 },
    { left: '66%', height: 110, opacity: 0.15, duration: 5.8, delay: 0.9 },
    { left: '78%', height: 200, opacity: 0.30, duration: 7.6, delay: 0.4 },
    { left: '90%', height: 140, opacity: 0.20, duration: 6.2, delay: 1.3 },
  ];

  document.querySelectorAll(".floating-trees").forEach((container) => {
    layout.forEach((t) => {
      const tree = document.createElement("div");
      tree.className = "tree";
      tree.style.left = t.left;
      tree.style.width = `${t.height * 0.7}px`;
      tree.style.height = `${t.height}px`;
      tree.style.opacity = t.opacity;
      tree.style.animationDuration = `${t.duration}s`;
      tree.style.animationDelay = `${t.delay}s`;
      tree.innerHTML = treeSVG;
      container.appendChild(tree);
    });
  });

});