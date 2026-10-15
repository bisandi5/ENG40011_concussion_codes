const startButton = document.getElementById("start-button");
const stopButton = document.getElementById("stop-button");
const statusText = document.getElementById("recording-status");
const elapsedText = document.getElementById("elapsed-time");

let timer = null;
let startTime = 0;

startButton.addEventListener("click", function () {
    startTime = Date.now();
    elapsedText.textContent = "0";
    statusText.textContent = "Recording demo";

    startButton.disabled = true;
    stopButton.disabled = false;

    timer = setInterval(function () {
        const seconds = Math.floor((Date.now() - startTime) / 1000);
        elapsedText.textContent = seconds;
    }, 200);
});

stopButton.addEventListener("click", function () {
    clearInterval(timer);
    timer = null;

    const seconds = Math.floor((Date.now() - startTime) / 1000);
    elapsedText.textContent = seconds;
    statusText.textContent = "Demo stopped";

    startButton.disabled = false;
    stopButton.disabled = true;
});

