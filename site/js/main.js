import { drawFigures } from './writeup.js';

const docEl = document.documentElement;
document.querySelector('.theme-toggle').addEventListener('click', () => {
  const next = docEl.dataset.theme === 'dark' ? 'light' : 'dark';
  docEl.dataset.theme = next;
  try { localStorage.setItem('cfe-theme', next); } catch (e) { /* storage unavailable: theme lasts this visit */ }
});

drawFigures(document.querySelector('.post'));
