// ==========================================
// static/script.js
// Modern AI Car Price Prediction
// ==========================================

console.log("AI Car Price Prediction Loaded");

/* ==========================================
FORM ANIMATION
========================================== */

const inputs = document.querySelectorAll("input");

inputs.forEach((input) => {

    input.addEventListener("focus", () => {

        input.style.boxShadow = "0 0 15px rgba(96,165,250,0.6)";

    });

    input.addEventListener("blur", () => {

        input.style.boxShadow = "none";

    });

});

/* ==========================================
BUTTON LOADING EFFECT
========================================== */

const form = document.querySelector("form");

const button = document.querySelector("button");

form.addEventListener("submit", () => {

    button.innerHTML =
        '<i class="fa-solid fa-spinner fa-spin"></i> Predicting...';

    button.disabled = true;

});

/* ==========================================
SMOOTH PAGE LOAD
========================================== */

window.addEventListener("load", () => {

    document.body.style.opacity = "1";

});

/* ==========================================
SCROLL ANIMATION
========================================== */

window.addEventListener("scroll", () => {

    const scrolled = window.scrollY;

    document.querySelector(".background").style.transform =
        `translateY(${scrolled * 0.2}px)`;

});